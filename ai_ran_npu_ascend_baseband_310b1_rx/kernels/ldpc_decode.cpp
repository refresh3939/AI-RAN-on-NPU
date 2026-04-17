/**
 * @file ldpc_decode.cpp — LDPC bit-flipping decoder + information bit recovery
 *
 * Three-stage processing:
 *   Stage 1 (Matmul mm1): Syndrome = codeword × H^T → parity check
 *   Stage 2 (Matmul mm2): Votes = syndrome^T × H → bit reliability
 *   Stage 3 (LdpcFlipping): Flip bits with max votes, iterate up to 20 times
 *   Stage 4 (Matmul mm3): info = decoded[:,:256] × G_left_inv → information bits
 *
 * GM args (9): a, b, c1, mask, c, g_inv, info_out, workspace, tilingGm
 *   a:        [M, 512] int8   — input codewords
 *   b:        [512, 256] int8 — parity check matrix H
 *   c1:       [M, 512] int8  — intermediate syndrome (reused for info extraction)
 *   mask:     [8] int32      — per-core convergence flags
 *   c:        [M, 512] int32 — vote accumulator
 *   g_inv:    [256, 256] int8 — left inverse of generator matrix
 *   info_out: [M, 256] int16 — decoded information bits
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize, GM_ADDR tilingGM) {
    auto ptr = reinterpret_cast<uint32_t*>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++)
        ptr[i] = src[i];
    localMemSize = *reinterpret_cast<__gm__ uint64_t*>(tilingGM + sizeof(TCubeTiling));
}

__aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling,
    int &offsetA, int &offsetB, int &offsetC,
    int &tailM, int &tailN, bool isTransA, bool isTransB)
{
    uint32_t mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    if (isTransA) offsetA = mCoreIndx * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    if (isTransB) offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

    tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

// ============================================================
// Bit-flipping engine: flips bits with maximum vote count
// ============================================================
template<typename T>
class LdpcFlipping {
public:
    __aicore__ inline LdpcFlipping() {}

    __aicore__ inline void Init(GM_ADDR votesGM, GM_ADDR bitsGM,
                                uint32_t totalRows, uint32_t rowLen,
                                AscendC::TPipe *pipe)
    {
        m_totalRows = totalRows;
        m_rowLen = rowLen;
        m_pipe = pipe;
        m_lenBurstAligned = AscendC::AlignUp(rowLen, 32);

        m_pipe->InitBuffer(Q_Votes, 2, m_lenBurstAligned * sizeof(int32_t));
        m_pipe->InitBuffer(Q_Bits, 2, m_lenBurstAligned * sizeof(int8_t));
        m_pipe->InitBuffer(Q_Out, 2, m_lenBurstAligned * sizeof(int8_t));
        m_pipe->InitBuffer(B_Calc, m_lenBurstAligned * sizeof(float));
        m_pipe->InitBuffer(B_Calc_Original, m_lenBurstAligned * sizeof(float));
        m_pipe->InitBuffer(reduceBuffer, m_lenBurstAligned * sizeof(float));
        m_pipe->InitBuffer(selMaskBuffer, 128);
        m_pipe->InitBuffer(B_Scalar, 128);

        bitsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(bitsGM), m_totalRows * m_rowLen);
        votesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(votesGM), m_totalRows * m_rowLen);
    }

    __aicore__ inline void SetMask(AscendC::LocalTensor<int16_t> maskTensor) {
        m_maskTensor = maskTensor;
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < m_totalRows; i++) {
            if (m_maskTensor.GetValue(i) == 0) continue;  // already converged
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t index) {
        auto votesLocal = Q_Votes.AllocTensor<int32_t>();
        auto bitsLocal = Q_Bits.AllocTensor<int8_t>();
        AscendC::DataCopy(votesLocal, votesGlobal[index * m_rowLen], m_lenBurstAligned);
        AscendC::DataCopy(bitsLocal, bitsGlobal[index * m_rowLen], m_lenBurstAligned);
        Q_Votes.EnQue(votesLocal);
        Q_Bits.EnQue(bitsLocal);
    }

    __aicore__ inline void Compute(uint32_t index) {
        auto votesLocal = Q_Votes.DeQue<int32_t>();
        auto bitsLocal = Q_Bits.DeQue<int8_t>();
        auto outLocal = Q_Out.AllocTensor<int8_t>();

        auto votesFloat = B_Calc_Original.Get<float>();
        auto maxValScalar = B_Scalar.Get<float>();
        auto maxValTensor = B_Calc.Get<float>();
        auto tempFloat = reduceBuffer.Get<float>();
        auto selMask = selMaskBuffer.Get<uint8_t>();

        // Find max vote
        AscendC::Cast(votesFloat, votesLocal, AscendC::RoundMode::CAST_NONE, m_rowLen);
        AscendC::ReduceMax(maxValScalar, votesFloat, tempFloat, m_rowLen, true);
        AscendC::PipeBarrier<PIPE_ALL>();

        float maxVal = maxValScalar.GetValue(0);
        AscendC::Duplicate(maxValTensor, maxVal, m_rowLen);
        AscendC::Compare(selMask, votesFloat, maxValTensor, AscendC::CMPMODE::EQ, m_rowLen);
        AscendC::PipeBarrier<PIPE_V>();

        // Build flip mask: 1 where votes == maxVotes and maxVotes > 0
        auto oneTensor = B_Calc_Original.Get<float>();
        auto zeroTensor = reduceBuffer.Get<float>();
        auto maskElement = B_Calc.Get<float>();

        AscendC::Duplicate(zeroTensor, 0.0f, m_rowLen);
        float fillValue = (maxVal > 0.5f) ? 1.0f : 0.0f;
        AscendC::Duplicate(oneTensor, fillValue, m_rowLen);
        AscendC::Select(maskElement, selMask, oneTensor, zeroTensor,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, m_rowLen);

        auto xor_mask = reduceBuffer.Get<int16_t>();
        AscendC::Cast(xor_mask, maskElement, AscendC::RoundMode::CAST_RINT, m_rowLen);
        AscendC::PipeBarrier<PIPE_V>();

        // XOR: flip selected bits
        auto tempHalf = B_Calc.Get<half>();
        auto xor_bits = B_Calc_Original.Get<int16_t>();
        AscendC::Cast(tempHalf, bitsLocal, AscendC::RoundMode::CAST_NONE, m_rowLen);
        AscendC::Cast(xor_bits, tempHalf, AscendC::RoundMode::CAST_RINT, m_rowLen);
        AscendC::PipeBarrier<PIPE_V>();

        auto xor_out = B_Calc.Get<int16_t>();
        auto xorTmp = selMaskBuffer.Get<uint8_t>();
        AscendC::Xor(xor_out, xor_bits, xor_mask, xorTmp, m_rowLen);
        AscendC::PipeBarrier<PIPE_V>();

        auto outHalf = reduceBuffer.Get<half>();
        AscendC::Cast(outHalf, xor_out, AscendC::RoundMode::CAST_NONE, m_rowLen);
        AscendC::Cast(outLocal, outHalf, AscendC::RoundMode::CAST_NONE, m_rowLen);

        Q_Out.EnQue(outLocal);
        Q_Votes.FreeTensor(votesLocal);
        Q_Bits.FreeTensor(bitsLocal);
    }

    __aicore__ inline void CopyOut(uint32_t index) {
        auto outLocal = Q_Out.DeQue<int8_t>();
        AscendC::DataCopy(bitsGlobal[index * m_rowLen], outLocal, m_lenBurstAligned);
        Q_Out.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe *m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> Q_Votes, Q_Bits;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> Q_Out;
    AscendC::TBuf<AscendC::TPosition::VECCALC> B_Calc, B_Calc_Original, reduceBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selMaskBuffer, B_Scalar;
    AscendC::LocalTensor<int16_t> m_maskTensor;
    AscendC::GlobalTensor<int8_t> bitsGlobal;
    AscendC::GlobalTensor<int32_t> votesGlobal;
    uint32_t m_totalRows, m_rowLen, m_lenBurstAligned;
};

// ============================================================
// Main LDPC decode kernel
// ============================================================
extern "C" __global__ __aicore__ void ldpc_decode(
    GM_ADDR a, GM_ADDR b, GM_ADDR c1, GM_ADDR mask, GM_ADDR c,
    GM_ADDR g_inv, GM_ADDR info_out,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = int8_t;
    using B_T = int8_t;
    using C_T = int32_t;

    AscendC::TPipe pipe;
    TCubeTiling tiling, tiling2, tiling3;
    uint64_t localMemSize = 0, localMemSize2 = 0, localMemSize3 = 0;

    CopyTiling(&tiling, localMemSize, tilingGm);
    CopyTiling(&tiling2, localMemSize2, tilingGm + 2048);
    CopyTiling(&tiling3, localMemSize3, tilingGm + 4096);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<int32_t> cGlobal, maskGlobal;
    AscendC::GlobalTensor<int8_t> c1Global, gInvGlobal;
    AscendC::GlobalTensor<int16_t> infoOutGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(b), tiling.Ka * tiling.N);
    maskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(mask) + AscendC::GetBlockIdx() * 8, 8);
    c1Global.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(c1), tiling.M * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(c), tiling2.M * tiling2.N);
    gInvGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(g_inv), 256 * 256);
    infoOutGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(info_out), tiling.M * 256);

    // GM offsets for Matmul 1 (syndrome)
    int offsetA = 0, offsetB = 0, offsetC = 0, tailM = 0, tailN = 0;
    CalcGMOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, false, false);
    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC1 = c1Global[offsetC];

    size_t batch_size = tiling.singleCoreM;

    AscendC::TBuf<AscendC::TPosition::A1> ABuf;
    pipe.InitBuffer(ABuf, tiling2.singleCoreM * tiling2.singleCoreK * sizeof(int8_t));

    AscendC::TBuf<AscendC::TPosition::VECCALC> cBuf;
    pipe.InitBuffer(cBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(C_T));
    AscendC::LocalTensor<C_T> cLocal = cBuf.Get<C_T>(tiling.singleCoreM * tiling.singleCoreN);
    AscendC::Duplicate<C_T>(cLocal, (int32_t)0, tiling.singleCoreM * tiling.singleCoreN);

    AscendC::TBuf<AscendC::TPosition::VECCALC> oneBuf;
    pipe.InitBuffer(oneBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int16_t));

    AscendC::TBuf<AscendC::TPosition::VECCALC> sBuf;
    pipe.InitBuffer(sBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int8_t));
    AscendC::LocalTensor<int8_t> sLocal = sBuf.Get<int8_t>();

    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf, scalarBuf, sharedTmpBuf;
    pipe.InitBuffer(maskBuf, batch_size * sizeof(int16_t));
    pipe.InitBuffer(scalarBuf, 64);
    pipe.InitBuffer(sharedTmpBuf, tiling.singleCoreN * sizeof(half));

    AscendC::TBuf<AscendC::TPosition::VECCALC> int16tmpBuf;
    pipe.InitBuffer(int16tmpBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int16_t));

    int16_t coreTotalMaskSum = 0;

    // Matmul objects
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, C_T>> mm;
    REGIST_MATMUL_OBJ(&pipe, workspace, mm, &tiling);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>> mm2;
    REGIST_MATMUL_OBJ(&pipe, workspace, mm2, &tiling2);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
           MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, int32_t>> mm3;
    REGIST_MATMUL_OBJ(&pipe, workspace, mm3, &tiling3);

    // GM offsets for Matmul 2 (votes)
    int offsetA2 = 0, offsetB2 = 0, offsetC2 = 0, tailM2 = 0, tailN2 = 0;
    CalcGMOffset(AscendC::GetBlockIdx(), tiling2, offsetA2, offsetB2, offsetC2, tailM2, tailN2, false, true);
    AscendC::GlobalTensor<int8_t> gmA2 = c1Global[offsetA2];
    AscendC::GlobalTensor<int8_t> gmB2 = bGlobal[offsetB2];
    auto gmC = cGlobal[offsetC2];

    auto votesPhy = const_cast<__gm__ int32_t*>(gmC.GetPhyAddr());
    GM_ADDR votesAddr = reinterpret_cast<__gm__ uint8_t*>(votesPhy);
    auto bitsPhy = const_cast<__gm__ int8_t*>(gmA.GetPhyAddr());
    GM_ADDR bitsAddr = reinterpret_cast<__gm__ uint8_t*>(bitsPhy);

    uint32_t rows = tiling2.singleCoreM;
    uint32_t rowLen = tiling2.singleCoreN;

    AscendC::PipeBarrier<PIPE_ALL>();
    LdpcFlipping<int8_t> flipper;
    flipper.Init(votesAddr, bitsAddr, rows, rowLen, &pipe);

    // ============================================================
    // Iterative bit-flipping loop (max 20 iterations)
    // ============================================================
    constexpr int MAX_ITER = 20;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        coreTotalMaskSum = 0;

        if (AscendC::GetBlockIdx() >= tiling.usedCoreNum) return;

        // Stage 1: Syndrome computation (codeword × H)
        mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
        mm.SetTensorA(gmA, false);
        mm.SetTensorB(gmB, false);
        mm.SetTail(tailM, tailN);
        mm.IterateAll(cLocal);
        mm.End();
        AscendC::PipeBarrier<PIPE_ALL>();

        // Mod-2 reduction and convergence check
        int dataLength = tiling.singleCoreM * tiling.singleCoreN;
        AscendC::LocalTensor<int16_t> int16dst = int16tmpBuf.Get<int16_t>();
        AscendC::LocalTensor<int16_t> oneTensor = oneBuf.Get<int16_t>();
        AscendC::Duplicate<int16_t>(oneTensor, (int16_t)1, dataLength);

        AscendC::Cast(int16dst, cLocal, AscendC::RoundMode::CAST_NONE, dataLength);
        AscendC::And(int16dst, int16dst, oneTensor, dataLength);

        AscendC::LocalTensor<half> halfDst = oneTensor.ReinterpretCast<half>();
        AscendC::Cast(halfDst, int16dst, AscendC::RoundMode::CAST_NONE, dataLength);
        AscendC::LocalTensor<int8_t> c_int8 = int16dst.ReinterpretCast<int8_t>();
        AscendC::Cast(c_int8, halfDst, AscendC::RoundMode::CAST_NONE, dataLength);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gmC1, c_int8, dataLength);

        // Per-row syndrome weight (convergence check)
        AscendC::LocalTensor<int16_t> maskTensor = maskBuf.Get<int16_t>();
        AscendC::LocalTensor<half> sharedTmpLocal = sharedTmpBuf.Get<half>();
        AscendC::LocalTensor<half> scalarSum = scalarBuf.Get<half>(2);
        for (uint32_t i = 0; i < batch_size; i++) {
            AscendC::ReduceSum<half>(scalarSum, halfDst[i * tiling.N], sharedTmpLocal, tiling.N);
            AscendC::PipeBarrier<PIPE_V>();
            int16_t val = (int16_t)scalarSum.GetValue(0);
            maskTensor.SetValue(i, val);
            coreTotalMaskSum += val;
        }

        // Stage 2: Vote computation (syndrome^T × H)
        mm2.SetOrgShape(tiling2.M, tiling2.N, tiling2.Ka, tiling2.Kb);
        mm2.SetTensorA(gmA2, false);
        mm2.SetTensorB(gmB2, true);
        mm2.SetTail(tailM2, tailN2);
        mm2.IterateAll(gmC);
        mm2.End();
        AscendC::PipeBarrier<PIPE_ALL>();

        // Stage 3: Bit flipping
        flipper.SetMask(maskTensor);
        flipper.Process();
        AscendC::PipeBarrier<PIPE_ALL>();

        if (coreTotalMaskSum == 0) break;  // all syndromes zero → converged
    }

    // Write convergence status
    cLocal.SetValue(0, (int32_t)coreTotalMaskSum);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(maskGlobal, cLocal, 8);

    // ============================================================
    // Stage 4: Information bit recovery — info = decoded[:,:256] × G_inv mod 2
    // ============================================================
    AscendC::PipeBarrier<PIPE_ALL>();

    // Extract first 256 columns of decoded codeword (row-by-row copy)
    for (uint32_t row = 0; row < batch_size; row++) {
        AscendC::DataCopy(sLocal[0], gmA[row * tiling.Ka], 256);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gmC1[row * 256], sLocal[0], 256);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // Matmul 3: decoded[:,:256] × G_left_inv → information bits
    int offsetA3 = 0, offsetB3 = 0, offsetC3 = 0, tailM3 = 0, tailN3 = 0;
    CalcGMOffset(AscendC::GetBlockIdx(), tiling3, offsetA3, offsetB3, offsetC3, tailM3, tailN3, false, false);

    mm3.SetOrgShape(tiling3.M, tiling3.N, tiling3.Ka, tiling3.Kb);
    mm3.SetTensorA(c1Global[offsetA3], false);
    mm3.SetTensorB(gInvGlobal[offsetB3], false);
    mm3.SetTail(tailM3, tailN3);
    mm3.IterateAll(cLocal);
    mm3.End();
    AscendC::PipeBarrier<PIPE_ALL>();

    // Mod-2 → output as int16
    {
        int dataLen = tiling3.singleCoreM * tiling3.singleCoreN;
        auto int16dst3 = int16tmpBuf.Get<int16_t>();
        auto oneTensor3 = oneBuf.Get<int16_t>();
        AscendC::Duplicate<int16_t>(oneTensor3, (int16_t)1, dataLen);
        AscendC::Cast(int16dst3, cLocal, AscendC::RoundMode::CAST_NONE, dataLen);
        AscendC::And(int16dst3, int16dst3, oneTensor3, dataLen);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(infoOutGlobal[offsetC3], int16dst3, dataLen);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
