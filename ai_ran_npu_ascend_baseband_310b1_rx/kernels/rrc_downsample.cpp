/**
 * @file rrc_downsample.cpp — RRC matched filter + downsample (Matmul Toeplitz)
 *
 * Y = X @ T    [1192, 1088] × [1088, 256] → [1192, 256]
 * Output written at stride=272, offset=16 to match FFT input layout.
 * First 16 columns (CP region) are pre-zeroed by host memset.
 *
 * GM args (7): rxReal, rxImag, filterCoeff, outReal, outImag, workspace, tilingGm
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

constexpr int32_t USE_CORES  = 8;
constexpr int32_t SYM_IN     = 1088;
constexpr int32_t N_FFT      = 256;
constexpr int32_t N_SYM      = 1192;
constexpr int32_t TILE_M     = 32;
constexpr int32_t OUT_STRIDE = 272;   // SYM_TD = N_FFT + NCP
constexpr int32_t OUT_OFFSET = 16;    // NCP: column offset for CP removal

class RrcDownDecpKernel {
public:
    __aicore__ inline RrcDownDecpKernel() {}

    __aicore__ inline void Init(
        GM_ADDR rxReal, GM_ADDR rxImag, GM_ADDR filterCoeff,
        GM_ADDR outReal, GM_ADDR outImag, GM_ADDR workspace,
        const TCubeTiling &cubeTiling, AscendC::TPipe *pipe)
    {
        this->pipe = pipe;
        this->cubeTiling = cubeTiling;

        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t perCore = N_SYM / USE_CORES, rem = N_SYM % USE_CORES;
        if (blockIdx < rem) {
            myStartRow = blockIdx * (perCore + 1); myRows = perCore + 1;
        } else {
            myStartRow = rem * (perCore + 1) + (blockIdx - rem) * perCore; myRows = perCore;
        }

        inRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rxReal), N_SYM * SYM_IN);
        inIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rxImag), N_SYM * SYM_IN);
        outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outReal), N_SYM * OUT_STRIDE);
        outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outImag), N_SYM * OUT_STRIDE);
        tGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(filterCoeff), SYM_IN * N_FFT);

        pipe->InitBuffer(mmWorkBuf, 32 * 1024);
        pipe->InitBuffer(mmResultBuf, TILE_M * N_FFT * sizeof(float));
        pipe->InitBuffer(outQ, 2, TILE_M * N_FFT * sizeof(half));

        REGIST_MATMUL_OBJ(pipe, workspace, mm, &(this->cubeTiling));
        AscendC::LocalTensor<uint8_t> mmWork = mmWorkBuf.Get<uint8_t>(32 * 1024);
        mm.SetLocalWorkspace(mmWork);
    }

    __aicore__ inline void Process() {
        DoMatmulPass(inRGm, outRGm);
        DoMatmulPass(inIGm, outIGm);
    }

private:
    __aicore__ inline void DoMatmulPass(
        AscendC::GlobalTensor<half> &inGm, AscendC::GlobalTensor<half> &outGm)
    {
        AscendC::LocalTensor<float> mmOut = mmResultBuf.Get<float>(TILE_M * N_FFT);
        int32_t processed = 0;

        while (processed < myRows) {
            int32_t tile = (myRows - processed >= TILE_M) ? TILE_M : (myRows - processed);

            mm.SetOrgShape(tile, N_FFT, SYM_IN, SYM_IN);
            mm.SetTensorA(inGm[(myStartRow + processed) * SYM_IN], false);
            mm.SetTensorB(tGm, false);
            mm.SetTail(tile, N_FFT);
            mm.IterateAll(mmOut);
            mm.End();

            AscendC::TQueSync<PIPE_M, PIPE_V> sync;
            sync.SetFlag(0); sync.WaitFlag(0);

            for (int32_t r = 0; r < tile; r++) {
                auto oLocal = outQ.AllocTensor<half>();
                AscendC::Cast(oLocal, mmOut[r * N_FFT], AscendC::RoundMode::CAST_NONE, N_FFT);
                outQ.EnQue(oLocal);
                oLocal = outQ.DeQue<half>();
                int32_t gmOff = (myStartRow + processed + r) * OUT_STRIDE + OUT_OFFSET;
                AscendC::DataCopy(outGm[gmOff], oLocal, N_FFT);
                outQ.FreeTensor(oLocal);
            }
            processed += tile;
        }
    }

private:
    AscendC::TPipe *pipe;
    TCubeTiling cubeTiling;
    int32_t myStartRow, myRows;
    AscendC::GlobalTensor<half> inRGm, inIGm, outRGm, outIGm, tGm;
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>> mm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmWorkBuf, mmResultBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
};

extern "C" __global__ __aicore__ void rrc_downsample(
    GM_ADDR rxReal, GM_ADDR rxImag, GM_ADDR filterCoeff,
    GM_ADDR outReal, GM_ADDR outImag, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling cubeTiling;
    auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
    auto ptr = reinterpret_cast<uint32_t*>(&cubeTiling);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++) ptr[i] = src[i];

    RrcDownDecpKernel kernel;
    kernel.Init(rxReal, rxImag, filterCoeff, outReal, outImag, workspace, cubeTiling, &pipe);
    kernel.Process();
}
