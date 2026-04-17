/**
 * @file fft_postproc.cpp — FFT post-processing + delay compensation
 *
 * Combines cos/sin DFT outputs into complex frequency-domain signal:
 *   freq_r = fft_cos_R - fft_sin_I
 *   freq_i = fft_cos_I + fft_sin_R
 * Then applies per-subcarrier delay compensation: freq *= dc (complex multiply).
 *
 * GM args (9): fft0, fft1, fft2, fft3, dcR, dcI, outR, outI, tiling
 *   fft0/1 = cos DFT output (R/I), fft2/3 = sin DFT output (R/I)
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t N_FFT      = 256;
constexpr int32_t BATCH_ROWS = 32;

class FftPostprocKernel {
public:
    __aicore__ inline FftPostprocKernel() {}

    __aicore__ inline void Init(GM_ADDR fft0, GM_ADDR fft1, GM_ADDR fft2, GM_ADDR fft3,
        GM_ADDR dcR, GM_ADDR dcI, GM_ADDR outR, GM_ADDR outI, GM_ADDR tilingGM)
    {
        auto tg = reinterpret_cast<__gm__ int32_t*>(tilingGM);
        int32_t totalRows = tg[0];

        int32_t coreId = GetBlockIdx();
        int32_t rowsPerCore = totalRows / 8;
        int32_t startRow = coreId * rowsPerCore;
        if (coreId == 7) rowsPerCore = totalRows - startRow;
        numRows = rowsPerCore;

        int32_t off = startRow * N_FFT;
        gmFft0.SetGlobalBuffer((__gm__ half*)fft0 + off, rowsPerCore * N_FFT);
        gmFft1.SetGlobalBuffer((__gm__ half*)fft1 + off, rowsPerCore * N_FFT);
        gmFft2.SetGlobalBuffer((__gm__ half*)fft2 + off, rowsPerCore * N_FFT);
        gmFft3.SetGlobalBuffer((__gm__ half*)fft3 + off, rowsPerCore * N_FFT);
        gmOutR.SetGlobalBuffer((__gm__ half*)outR + off, rowsPerCore * N_FFT);
        gmOutI.SetGlobalBuffer((__gm__ half*)outI + off, rowsPerCore * N_FFT);
        gmDcR.SetGlobalBuffer((__gm__ half*)dcR, N_FFT);
        gmDcI.SetGlobalBuffer((__gm__ half*)dcI, N_FFT);

        int32_t batchElems = BATCH_ROWS * N_FFT;
        pipe.InitBuffer(bufF0, batchElems * 2); pipe.InitBuffer(bufF1, batchElems * 2);
        pipe.InitBuffer(bufF2, batchElems * 2); pipe.InitBuffer(bufF3, batchElems * 2);
        pipe.InitBuffer(bufDcR, N_FFT * 2);     pipe.InitBuffer(bufDcI, N_FFT * 2);
        pipe.InitBuffer(bufFreqR, batchElems * 2); pipe.InitBuffer(bufFreqI, batchElems * 2);
        pipe.InitBuffer(bufTmpA, batchElems * 2);  pipe.InitBuffer(bufTmpB, batchElems * 2);
    }

    __aicore__ inline void Process() {
        auto dcR = bufDcR.Get<half>(), dcI = bufDcI.Get<half>();
        DataCopy(dcR, gmDcR, N_FFT);
        DataCopy(dcI, gmDcI, N_FFT);
        PipeBarrier<PIPE_ALL>();

        int32_t done = 0;
        while (done < numRows) {
            int32_t br = (numRows - done >= BATCH_ROWS) ? BATCH_ROWS : (numRows - done);
            int32_t el = br * N_FFT, go = done * N_FFT;

            auto f0 = bufF0.Get<half>(), f1 = bufF1.Get<half>();
            auto f2 = bufF2.Get<half>(), f3 = bufF3.Get<half>();
            DataCopy(f0, gmFft0[go], el); DataCopy(f1, gmFft1[go], el);
            DataCopy(f2, gmFft2[go], el); DataCopy(f3, gmFft3[go], el);
            PipeBarrier<PIPE_ALL>();

            // Combine: freqR = cosR - sinI, freqI = cosI + sinR
            auto freqR = bufFreqR.Get<half>(), freqI = bufFreqI.Get<half>();
            Sub(freqR, f0, f3, el);
            Add(freqI, f1, f2, el);
            PipeBarrier<PIPE_ALL>();

            // Delay compensation: out = freq * dc (complex multiply)
            auto tmpA = bufTmpA.Get<half>(), tmpB = bufTmpB.Get<half>();
            for (int32_t r = 0; r < br; r++) {
                int32_t o = r * N_FFT;
                Mul(tmpA[o], freqR[o], dcR, N_FFT);
                Mul(tmpB[o], freqI[o], dcI, N_FFT);
            }
            PipeBarrier<PIPE_ALL>();
            Sub(f0, tmpA, tmpB, el);
            PipeBarrier<PIPE_ALL>();

            for (int32_t r = 0; r < br; r++) {
                int32_t o = r * N_FFT;
                Mul(tmpA[o], freqR[o], dcI, N_FFT);
                Mul(tmpB[o], freqI[o], dcR, N_FFT);
            }
            PipeBarrier<PIPE_ALL>();
            Add(f1, tmpA, tmpB, el);
            PipeBarrier<PIPE_ALL>();

            DataCopy(gmOutR[go], f0, el);
            DataCopy(gmOutI[go], f1, el);
            PipeBarrier<PIPE_ALL>();
            done += br;
        }
    }

private:
    int32_t numRows;
    GlobalTensor<half> gmFft0, gmFft1, gmFft2, gmFft3, gmDcR, gmDcI, gmOutR, gmOutI;
    TBuf<TPosition::VECCALC> bufF0, bufF1, bufF2, bufF3;
    TBuf<TPosition::VECCALC> bufDcR, bufDcI, bufFreqR, bufFreqI, bufTmpA, bufTmpB;
    TPipe pipe;
};

extern "C" __global__ __aicore__ void fft_postproc(
    GM_ADDR fft0, GM_ADDR fft1, GM_ADDR fft2, GM_ADDR fft3,
    GM_ADDR dcR, GM_ADDR dcI, GM_ADDR outR, GM_ADDR outI, GM_ADDR tiling)
{
    FftPostprocKernel op;
    op.Init(fft0, fft1, fft2, fft3, dcR, dcI, outR, outI, tiling);
    op.Process();
}
