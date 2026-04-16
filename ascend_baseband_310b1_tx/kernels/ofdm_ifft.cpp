/**
 * @file ofdm_ifft.cpp — OFDM IDFT via Matmul (cos/sin decomposition)
 *
 * Computes a single-sided DFT matrix product as two real Matmul ops:
 *   out0 = (X_real × B^T) [+ bias, when bias term present in workspace]
 *   out1 =  X_imag × B^T
 *
 * Called twice from the host: once with cos basis (bias = pilot_real),
 * once with sin basis (bias = pilot_imag). The host combines outputs via
 * the ifft_postproc kernel.
 *
 * Input stride K_IN=224 is padded to K_PAD=272 inside the kernel via a
 * per-core pad workspace region to satisfy Matmul K-dim alignment.
 *
 * GM args (8): qamR, qamI, out0, out1, matB, matBdummy, workspace, matmulWs
 *   matmulWs is an explicit dedicated workspace (not GetSysWorkSpacePtr)
 *   to avoid conflicts when launched multiple times on the same stream.
 *
 * Workspace layout (half offsets):
 *   [0 .. 511]   TCubeTiling
 *   [512 .. 784] bias vector (pilot_real or pilot_imag, 272 half)
 *   [2M+blk*TILE .. ] per-core pad region (padded Matmul A input)
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

constexpr int32_t USE_CORES = 8;
constexpr int32_t M_TOTAL   = 1192;
constexpr int32_t K_IN      = 224;
constexpr int32_t K_PAD     = 272;
constexpr int32_t N_FULL    = 272;
constexpr int32_t N_TILE    = 256;
constexpr int32_t TILE_M    = 32;

constexpr int32_t BIAS_OFFSET_HALF = 512;
constexpr int32_t PAD_GM_OFFSET    = 4 * 1024 * 1024 / 2;
constexpr int32_t PAD_TILE_SIZE    = TILE_M * K_PAD;
constexpr float   SCALE            = 1.0f / 256.0f;

class OfdmIfftKernel {
public:
    __aicore__ inline OfdmIfftKernel() {}

    __aicore__ inline void Init(
        GM_ADDR qamR, GM_ADDR qamI, GM_ADDR out0, GM_ADDR out1,
        GM_ADDR matB, GM_ADDR workspace, GM_ADDR matmulWs,
        const TCubeTiling &cubeTiling, AscendC::TPipe *pipe)
    {
        this->pipe = pipe;
        this->cubeTiling = cubeTiling;

        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t perCore = M_TOTAL / USE_CORES;
        int32_t rem = M_TOTAL % USE_CORES;
        if (blockIdx < rem) {
            myStartRow = blockIdx * (perCore + 1);
            myRows = perCore + 1;
        } else {
            myStartRow = rem * (perCore + 1) + (blockIdx - rem) * perCore;
            myRows = perCore;
        }

        aRGm.SetGlobalBuffer((__gm__ half*)qamR, M_TOTAL * K_IN);
        aIGm.SetGlobalBuffer((__gm__ half*)qamI, M_TOTAL * K_IN);
        out0Gm.SetGlobalBuffer((__gm__ half*)out0, M_TOTAL * N_FULL);
        out1Gm.SetGlobalBuffer((__gm__ half*)out1, M_TOTAL * N_FULL);
        bGm.SetGlobalBuffer((__gm__ half*)matB, N_FULL * K_PAD);

        auto wsH = reinterpret_cast<__gm__ half*>(workspace);
        biasGm.SetGlobalBuffer(wsH + BIAS_OFFSET_HALF, N_FULL);
        padGm.SetGlobalBuffer(wsH + PAD_GM_OFFSET + blockIdx * PAD_TILE_SIZE,
                              PAD_TILE_SIZE);

        pipe->InitBuffer(mmWorkBuf,    32 * 1024);
        pipe->InitBuffer(mmResultBuf,  TILE_M * N_TILE * sizeof(float));
        pipe->InitBuffer(outQ, 2,       TILE_M * N_TILE * sizeof(half));
        pipe->InitBuffer(biasBuf,      N_FULL * sizeof(half));
        pipe->InitBuffer(padUB,        TILE_M * K_IN * sizeof(half));

        REGIST_MATMUL_OBJ(pipe, matmulWs, mm, &(this->cubeTiling));
        mm.SetLocalWorkspace(mmWorkBuf.Get<uint8_t>(32 * 1024));

        // Preload bias vector
        auto bias = biasBuf.Get<half>(N_FULL);
        AscendC::DataCopy(bias, biasGm, N_FULL);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process() {
        if (myRows <= 0) return;
        DoMatmul(aRGm, out0Gm, true);   // real part with bias
        DoMatmul(aIGm, out1Gm, false);  // imag part without bias
    }

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>> mm;
    TCubeTiling cubeTiling;

private:
    __aicore__ inline void PadInput(
        AscendC::GlobalTensor<half> &inGm, int32_t startRow, int32_t tile)
    {
        auto pBuf = padUB.Get<half>(TILE_M * K_IN);
        int32_t totalElems = tile * K_IN;
        int32_t totalPad = ((totalElems + 15) / 16) * 16;
        AscendC::DataCopy(pBuf, inGm[startRow * K_IN], totalPad);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopyParams params;
        params.blockCount = tile;
        params.blockLen  = K_IN * sizeof(half) / AscendC::DEFAULT_C0_SIZE;
        params.srcStride = 0;
        params.dstStride = (K_PAD - K_IN) * sizeof(half) / AscendC::DEFAULT_C0_SIZE;
        AscendC::DataCopy(padGm, pBuf, params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void DoMatmul(
        AscendC::GlobalTensor<half> &inGm, AscendC::GlobalTensor<half> &outGm,
        bool addBias)
    {
        int32_t processed = 0;
        while (processed < myRows) {
            int32_t tile = (myRows - processed >= TILE_M) ? TILE_M
                                                           : (myRows - processed);
            PadInput(inGm, myStartRow + processed, tile);
            for (int32_t nOff = 0; nOff < N_FULL; nOff += N_TILE) {
                int32_t nTile = (nOff + N_TILE > N_FULL) ? (N_FULL - nOff) : N_TILE;
                int32_t nTilePad = ((nTile + 15) / 16) * 16;
                DoMatmulNTile(outGm, nOff, nTile, nTilePad, processed, tile, addBias);
            }
            processed += tile;
        }
    }

    __aicore__ inline void DoMatmulNTile(
        AscendC::GlobalTensor<half> &outGm,
        int32_t nOff, int32_t nTile, int32_t nTilePad,
        int32_t processed, int32_t tile, bool addBias)
    {
        auto mmOut = mmResultBuf.Get<float>(TILE_M * N_TILE);
        auto bias  = biasBuf.Get<half>(N_FULL);

        mm.SetOrgShape(tile, nTilePad, K_PAD);
        mm.SetTensorA(padGm, false);
        mm.SetTensorB(bGm[nOff * K_PAD], true);
        mm.SetTail(tile, nTile);
        while (mm.template Iterate<true>())
            mm.template GetTensorC<true>(mmOut, false, true);
        mm.End();

        AscendC::TQueSync<PIPE_M, PIPE_V> sync;
        sync.SetFlag(0); sync.WaitFlag(0);

        AscendC::Muls(mmOut, mmOut, SCALE, tile * nTilePad);
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t r = 0; r < tile; r++) {
            auto outLocal = outQ.AllocTensor<half>();
            AscendC::Cast(outLocal, mmOut[r * nTilePad],
                          AscendC::RoundMode::CAST_NONE, nTilePad);
            if (addBias) {
                AscendC::Add(outLocal, outLocal, bias[nOff], nTilePad);
            }
            outQ.EnQue(outLocal);
            outLocal = outQ.DeQue<half>();
            int32_t gmOff = (myStartRow + processed + r) * N_FULL + nOff;
            AscendC::DataCopy(outGm[gmOff], outLocal, nTilePad);
            outQ.FreeTensor(outLocal);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe *pipe;
    int32_t myStartRow, myRows;
    AscendC::GlobalTensor<half> aRGm, aIGm, out0Gm, out1Gm, bGm, biasGm, padGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmWorkBuf, mmResultBuf, biasBuf, padUB;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
};

extern "C" __global__ __aicore__ void ofdm_ifft(
    GM_ADDR qamR, GM_ADDR qamI, GM_ADDR out0, GM_ADDR out1,
    GM_ADDR matB, GM_ADDR matBdummy, GM_ADDR workspace, GM_ADDR matmulWs)
{
    (void)matBdummy;
    AscendC::TPipe pipe;
    TCubeTiling cubeTiling;
    {
        auto src = reinterpret_cast<__gm__ uint32_t*>(workspace);
        uint32_t *ptr = reinterpret_cast<uint32_t*>(&cubeTiling);
        for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++)
            ptr[i] = src[i];
    }
    OfdmIfftKernel kernel;
    kernel.Init(qamR, qamI, out0, out1, matB, workspace, matmulWs, cubeTiling, &pipe);
    kernel.Process();
}
