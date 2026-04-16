/**
 * @file ldpc_encode.cpp — LDPC encoder (GF(2) Matmul)
 *
 * Computes c = (m · G) mod 2 where m is the information bit matrix,
 * G is the generator matrix, and c is the codeword output.
 *
 * Implementation:
 *   1. int8 × int8 → int32 Matmul
 *   2. int32 → int16 Cast
 *   3. Bit-AND with 1 for mod-2 reduction
 *
 * GM args (5): a (info bits), b (gen matrix G), c (codewords),
 *              workspace, tiling
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

constexpr int16_t MOD2_MASK = 1;

__aicore__ inline uint32_t LdpcCeilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

__aicore__ inline void LdpcCopyTiling(TCubeTiling *tiling, uint64_t &localMemSize,
                                       GM_ADDR tilingGM)
{
    uint32_t *dst = reinterpret_cast<uint32_t*>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGM);
    constexpr uint32_t wordCount = sizeof(TCubeTiling) / sizeof(uint32_t);
    for (uint32_t i = 0; i < wordCount; i++) dst[i] = src[i];
    localMemSize = *reinterpret_cast<__gm__ uint64_t*>(tilingGM + sizeof(TCubeTiling));
}

__aicore__ inline void LdpcCalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                       int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                       int32_t &tailM, int32_t &tailN)
{
    uint32_t mBlockCount = LdpcCeilDiv(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIdx = blockIdx % mBlockCount;
    uint32_t nCoreIdx = blockIdx / mBlockCount;

    offsetA = mCoreIdx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIdx * tiling.singleCoreN;
    offsetC = mCoreIdx * tiling.N * tiling.singleCoreM + nCoreIdx * tiling.singleCoreN;

    int32_t remainM = tiling.M - mCoreIdx * tiling.singleCoreM;
    tailM = (remainM < tiling.singleCoreM) ? remainM : tiling.singleCoreM;
    int32_t remainN = tiling.N - nCoreIdx * tiling.singleCoreN;
    tailN = (remainN < tiling.singleCoreN) ? remainN : tiling.singleCoreN;
}

class LdpcEncodeKernel {
public:
    using A_TYPE   = int8_t;
    using B_TYPE   = int8_t;
    using C_TYPE   = int32_t;
    using OUT_TYPE = int16_t;

    __aicore__ inline LdpcEncodeKernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                GM_ADDR workspace, GM_ADDR tilingGM)
    {
        wsPtr_ = workspace;
        LdpcCopyTiling(&tiling_, localMemSize_, tilingGM);

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ A_TYPE*>(a), tiling_.M * tiling_.Ka);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ B_TYPE*>(b), tiling_.Ka * tiling_.N);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_TYPE*>(c), tiling_.M * tiling_.N);

        uint32_t tileSize = tiling_.singleCoreM * tiling_.singleCoreN;
        pipe_.InitBuffer(matmulBuf_, tileSize * sizeof(C_TYPE));
        pipe_.InitBuffer(castBuf_,   tileSize * sizeof(OUT_TYPE));
        pipe_.InitBuffer(outputBuf_, tileSize * sizeof(OUT_TYPE));
        pipe_.InitBuffer(maskBuf_,   tileSize * sizeof(OUT_TYPE));

        // Fill mask with 1s for mod-2 AND
        AscendC::LocalTensor<OUT_TYPE> mask = maskBuf_.Get<OUT_TYPE>();
        AscendC::Duplicate(mask, (OUT_TYPE)MOD2_MASK, tileSize);

        LdpcCalcOffset(AscendC::GetBlockIdx(), tiling_,
                       offsetA_, offsetB_, offsetC_, tailM_, tailN_);
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() >= tiling_.usedCoreNum) return;

        Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_TYPE>,
               MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_TYPE>,
               MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, C_TYPE>> mm;

        REGIST_MATMUL_OBJ(&pipe_, wsPtr_, mm, &tiling_);

        auto mmOut = matmulBuf_.Get<C_TYPE>();
        auto castOut = castBuf_.Get<OUT_TYPE>();
        auto out = outputBuf_.Get<OUT_TYPE>();
        auto mask = maskBuf_.Get<OUT_TYPE>();

        mm.SetOrgShape(tiling_.M, tiling_.N, tiling_.Ka, tiling_.Kb);
        mm.SetTensorA(aGm_[offsetA_], false);
        mm.SetTensorB(bGm_[offsetB_], false);
        mm.SetTail(tailM_, tailN_);
        mm.IterateAll(mmOut);
        mm.End();
        AscendC::PipeBarrier<PIPE_ALL>();

        uint32_t len = tiling_.singleCoreM * tiling_.singleCoreN;
        AscendC::Cast(castOut, mmOut, AscendC::RoundMode::CAST_NONE, len);
        AscendC::And(out, castOut, mask, len);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(cGm_[offsetC_], out, len);
    }

private:
    TCubeTiling tiling_;
    uint64_t localMemSize_ = 0;
    GM_ADDR wsPtr_ = nullptr;
    AscendC::GlobalTensor<A_TYPE>   aGm_;
    AscendC::GlobalTensor<B_TYPE>   bGm_;
    AscendC::GlobalTensor<OUT_TYPE> cGm_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matmulBuf_, maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECOUT>  castBuf_, outputBuf_;
    int32_t offsetA_ = 0, offsetB_ = 0, offsetC_ = 0;
    int32_t tailM_   = 0, tailN_   = 0;
};

extern "C" __global__ __aicore__ void ldpc_encode(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    LdpcEncodeKernel op;
    op.Init(a, b, c, workspace, tiling);
    op.Process();
}