/**
 * @file extract_subcarriers.cpp — Subcarrier extraction by index list
 *
 * Two modes:
 *   mode=0 (data): Extracts K subcarriers into separate R/I outputs
 *     dst_R/I[row, k] = src_R/I[row, scList[k]]
 *   mode=1 (pilot): Extracts K subcarriers into interleaved R/I output
 *     dst[row, 2k] = srcR[row, scList[k]], dst[row, 2k+1] = srcI[row, scList[k]]
 *
 * Note: For data extraction, the Matmul-based data_extract_mm kernel is faster.
 * This kernel is still used for pilot extraction (K=16, mode=1).
 *
 * GM args (7): srcR, srcI, scList, dstA, dstB, dummy, tiling
 * Tiling: {totalRows, nFFT, K, mode, padK}
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t N_FFT     = 256;
constexpr int32_t MAX_K     = 224;
constexpr int32_t USE_CORES = 8;

struct ExtractScTiling {
    int32_t totalRows;
    int32_t nFFT;
    int32_t K;
    int32_t mode;
    int32_t padK;
};

class ExtractSubcarriersKernel {
public:
    __aicore__ inline ExtractSubcarriersKernel() {}

    __aicore__ inline void Init(
        GM_ADDR srcR, GM_ADDR srcI, GM_ADDR scList,
        GM_ADDR dstA, GM_ADDR dstB, GM_ADDR dummy, GM_ADDR tilingGM)
    {
        auto tg = reinterpret_cast<__gm__ int32_t*>(tilingGM);
        int32_t totalRows = tg[0];
        K = tg[2];
        mode = tg[3];
        padK = tg[4];
        scAligned = (K + 7) & ~7;

        int32_t coreId = GetBlockIdx();
        int32_t rowsPerCore = totalRows / USE_CORES;
        startRow = coreId * rowsPerCore;
        if (coreId == USE_CORES - 1) rowsPerCore = totalRows - startRow;
        numRows = rowsPerCore;

        gmSrcR.SetGlobalBuffer((__gm__ half*)srcR, totalRows * N_FFT);
        gmSrcI.SetGlobalBuffer((__gm__ half*)srcI, totalRows * N_FFT);
        gmScList.SetGlobalBuffer((__gm__ int32_t*)scList, scAligned);

        if (mode == 0) {
            gmDstR.SetGlobalBuffer((__gm__ half*)dstA, totalRows * padK);
            gmDstI.SetGlobalBuffer((__gm__ half*)dstB, totalRows * padK);
        } else {
            gmDstRI.SetGlobalBuffer((__gm__ half*)dstA, totalRows * K * 2);
        }

        pipe.InitBuffer(bufSc, scAligned * sizeof(int32_t));
        pipe.InitBuffer(bufSrcR, N_FFT * sizeof(half));
        pipe.InitBuffer(bufSrcI, N_FFT * sizeof(half));
        if (mode == 0) {
            pipe.InitBuffer(bufDstR, padK * sizeof(half));
            pipe.InitBuffer(bufDstI, padK * sizeof(half));
        } else {
            pipe.InitBuffer(bufDstRI, K * 2 * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        LocalTensor<int32_t> scBuf = bufSc.Get<int32_t>();
        DataCopy(scBuf, gmScList, scAligned);
        PipeBarrier<PIPE_ALL>();

        for (int32_t r = 0; r < numRows; r++) {
            int32_t absRow = startRow + r;
            int32_t srcOff = absRow * N_FFT;

            LocalTensor<half> rowR = bufSrcR.Get<half>();
            LocalTensor<half> rowI = bufSrcI.Get<half>();
            DataCopy(rowR, gmSrcR[srcOff], N_FFT);
            DataCopy(rowI, gmSrcI[srcOff], N_FFT);
            PipeBarrier<PIPE_ALL>();

            if (mode == 0) {
                LocalTensor<half> dR = bufDstR.Get<half>();
                LocalTensor<half> dI = bufDstI.Get<half>();
                for (int32_t k = 0; k < K; k++) {
                    int32_t sc = scBuf.GetValue(k);
                    dR.SetValue(k, rowR.GetValue(sc));
                    dI.SetValue(k, rowI.GetValue(sc));
                }
                for (int32_t k = K; k < padK; k++) {
                    dR.SetValue(k, (half)0.0f);
                    dI.SetValue(k, (half)0.0f);
                }
                PipeBarrier<PIPE_ALL>();
                DataCopy(gmDstR[absRow * padK], dR, padK);
                DataCopy(gmDstI[absRow * padK], dI, padK);
            } else {
                LocalTensor<half> dRI = bufDstRI.Get<half>();
                for (int32_t k = 0; k < K; k++) {
                    int32_t sc = scBuf.GetValue(k);
                    dRI.SetValue(2 * k, rowR.GetValue(sc));
                    dRI.SetValue(2 * k + 1, rowI.GetValue(sc));
                }
                PipeBarrier<PIPE_ALL>();
                DataCopy(gmDstRI[absRow * K * 2], dRI, K * 2);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    int32_t startRow, numRows, K, mode, padK, scAligned;
    GlobalTensor<half> gmSrcR, gmSrcI, gmDstR, gmDstI, gmDstRI;
    GlobalTensor<int32_t> gmScList;
    TBuf<TPosition::VECCALC> bufSc, bufSrcR, bufSrcI, bufDstR, bufDstI, bufDstRI;
    TPipe pipe;
};

extern "C" __global__ __aicore__ void extract_subcarriers(
    GM_ADDR srcR, GM_ADDR srcI, GM_ADDR scList,
    GM_ADDR dstA, GM_ADDR dstB, GM_ADDR dummy, GM_ADDR tiling)
{
    ExtractSubcarriersKernel op;
    op.Init(srcR, srcI, scList, dstA, dstB, dummy, tiling);
    op.Process();
}
