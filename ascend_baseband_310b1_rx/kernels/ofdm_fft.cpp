/**
 * @file ofdm_fft.cpp — OFDM DFT via Matmul (fused CP removal + DFT)
 *
 * Computes frequency-domain output: out = ofdm_time × DFT_matrix^T
 *   Input:  [1192, 272] half (time-domain with CP, stride=272)
 *   DFT:    [256, 272] half (cos or sin component)
 *   Output: [1192, 256] half
 *
 * Called twice: once for cos (→ out0/out1), once for sin (→ out2/out3).
 * Combined by fft_postproc to form complex frequency-domain signal.
 *
 * GM args (8): ofdmR, ofdmI, out0, out1, matB, matBdummy, workspace, tilingGm
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
constexpr int32_t K         = 272;
constexpr int32_t N         = 256;
constexpr int32_t TILE_M    = 32;

class OfdmFftKernel {
public:
    __aicore__ inline OfdmFftKernel() {}

    __aicore__ inline void Init(
        GM_ADDR ofdmR, GM_ADDR ofdmI, GM_ADDR out0, GM_ADDR out1,
        GM_ADDR matB, GM_ADDR matBdummy, GM_ADDR workspace,
        const TCubeTiling &cubeTiling, AscendC::TPipe *pipe)
    {
        this->pipe = pipe;
        this->cubeTiling = cubeTiling;

        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t perCore = M_TOTAL / USE_CORES, rem = M_TOTAL % USE_CORES;
        if (blockIdx < rem) {
            myStartRow = blockIdx * (perCore + 1); myRows = perCore + 1;
        } else {
            myStartRow = rem * (perCore + 1) + (blockIdx - rem) * perCore; myRows = perCore;
        }

        aRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ofdmR), M_TOTAL * K);
        aIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ofdmI), M_TOTAL * K);
        out0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out0), M_TOTAL * N);
        out1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out1), M_TOTAL * N);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(matB), N * K);

        pipe->InitBuffer(mmWorkBuf, 32 * 1024);
        pipe->InitBuffer(mmResultBuf, TILE_M * N * (int32_t)sizeof(float));
        pipe->InitBuffer(outQ, 2, TILE_M * N * (int32_t)sizeof(half));

        REGIST_MATMUL_OBJ(pipe, workspace, mm, &(this->cubeTiling));
        AscendC::LocalTensor<uint8_t> mmWork = mmWorkBuf.Get<uint8_t>(32 * 1024);
        mm.SetLocalWorkspace(mmWork);
    }

    __aicore__ inline void Process() {
        if (myRows <= 0) return;
        DoMatmulToGm(aRGm, out0Gm);
        DoMatmulToGm(aIGm, out1Gm);
    }

private:
    __aicore__ inline void DoMatmulToGm(
        AscendC::GlobalTensor<half> &inGm, AscendC::GlobalTensor<half> &outGm)
    {
        AscendC::LocalTensor<float> mmOut = mmResultBuf.Get<float>(TILE_M * N);
        int32_t processed = 0;

        while (processed < myRows) {
            int32_t tile = (myRows - processed >= TILE_M) ? TILE_M : (myRows - processed);

            mm.SetOrgShape(tile, N, K);
            mm.SetTensorA(inGm[(myStartRow + processed) * K], false);
            mm.SetTensorB(bGm, true);
            mm.SetTail(tile, N);

            while (mm.template Iterate<true>()) {
                mm.template GetTensorC<true>(mmOut, false, true);
            }
            mm.End();

            AscendC::TQueSync<PIPE_M, PIPE_V> sync;
            sync.SetFlag(0); sync.WaitFlag(0);

            for (int32_t r = 0; r < tile; r++) {
                auto outLocal = outQ.AllocTensor<half>();
                AscendC::Cast(outLocal, mmOut[r * N], AscendC::RoundMode::CAST_NONE, N);
                outQ.EnQue(outLocal);
                outLocal = outQ.DeQue<half>();
                AscendC::DataCopy(outGm[(myStartRow + processed + r) * N], outLocal, N);
                outQ.FreeTensor(outLocal);
            }
            processed += tile;
        }
    }

private:
    AscendC::TPipe *pipe;
    TCubeTiling cubeTiling;
    int32_t myStartRow, myRows;
    AscendC::GlobalTensor<half> aRGm, aIGm, out0Gm, out1Gm, bGm;
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>> mm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmWorkBuf, mmResultBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
};

extern "C" __global__ __aicore__ void ofdm_fft(
    GM_ADDR ofdmR, GM_ADDR ofdmI, GM_ADDR out0, GM_ADDR out1,
    GM_ADDR matB, GM_ADDR matBdummy, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling cubeTiling;
    auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
    auto ptr = reinterpret_cast<uint32_t*>(&cubeTiling);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++) ptr[i] = src[i];

    OfdmFftKernel kernel;
    kernel.Init(ofdmR, ofdmI, out0, out1, matB, matBdummy, workspace, cubeTiling, &pipe);
    kernel.Process();
}
