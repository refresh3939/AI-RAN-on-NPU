/**
 * @file ifft_postproc.cpp — IFFT post-processing (cos/sin → complex)
 *
 * Combines the two Matmul outputs from ofdm_ifft into a complex time-domain
 * signal:
 *   td_r = out0 - out3   (cos×real − sin×imag)
 *   td_i = out1 + out2   (cos×imag + sin×real)
 *
 * Width = 272 (includes cyclic prefix). Unlike the RX version, no per-subcarrier
 * delay compensation is applied.
 *
 * GM args (7): out0, out1, out2, out3, outR, outI, tiling
 * Tiling: { totalRows (int32), nWidth (int32) } — nWidth currently unused
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t N_WIDTH    = 272;
constexpr int32_t BATCH_ROWS = 32;
constexpr int32_t USE_CORES  = 8;

class IfftPostprocKernel {
public:
    __aicore__ inline IfftPostprocKernel() {}

    __aicore__ inline void Init(GM_ADDR o0, GM_ADDR o1, GM_ADDR o2, GM_ADDR o3,
                                GM_ADDR outR, GM_ADDR outI, GM_ADDR tilingGM)
    {
        auto tg = reinterpret_cast<__gm__ int32_t*>(tilingGM);
        int32_t totalRows = tg[0];

        int32_t coreId = GetBlockIdx();
        int32_t perCore = totalRows / USE_CORES;
        int32_t startRow = coreId * perCore;
        if (coreId == USE_CORES - 1) perCore = totalRows - startRow;
        numRows_ = perCore;

        int32_t off = startRow * N_WIDTH;
        int32_t sz  = perCore * N_WIDTH;
        gmO0_.SetGlobalBuffer((__gm__ half*)o0 + off, sz);
        gmO1_.SetGlobalBuffer((__gm__ half*)o1 + off, sz);
        gmO2_.SetGlobalBuffer((__gm__ half*)o2 + off, sz);
        gmO3_.SetGlobalBuffer((__gm__ half*)o3 + off, sz);
        gmOutR_.SetGlobalBuffer((__gm__ half*)outR + off, sz);
        gmOutI_.SetGlobalBuffer((__gm__ half*)outI + off, sz);

        int32_t batchElems = BATCH_ROWS * N_WIDTH;
        pipe_.InitBuffer(bufO0_, batchElems * sizeof(half));
        pipe_.InitBuffer(bufO1_, batchElems * sizeof(half));
        pipe_.InitBuffer(bufO2_, batchElems * sizeof(half));
        pipe_.InitBuffer(bufO3_, batchElems * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t done = 0;
        while (done < numRows_) {
            int32_t br = (numRows_ - done >= BATCH_ROWS) ? BATCH_ROWS
                                                         : (numRows_ - done);
            int32_t el = br * N_WIDTH;
            int32_t go = done * N_WIDTH;

            auto f0 = bufO0_.Get<half>();
            auto f1 = bufO1_.Get<half>();
            auto f2 = bufO2_.Get<half>();
            auto f3 = bufO3_.Get<half>();

            DataCopy(f0, gmO0_[go], el);
            DataCopy(f1, gmO1_[go], el);
            DataCopy(f2, gmO2_[go], el);
            DataCopy(f3, gmO3_[go], el);
            PipeBarrier<PIPE_ALL>();

            Sub(f0, f0, f3, el);   // td_r = out0 - out3
            Add(f1, f1, f2, el);   // td_i = out1 + out2
            PipeBarrier<PIPE_V>();

            DataCopy(gmOutR_[go], f0, el);
            DataCopy(gmOutI_[go], f1, el);
            PipeBarrier<PIPE_ALL>();

            done += br;
        }
    }

private:
    int32_t numRows_;
    GlobalTensor<half> gmO0_, gmO1_, gmO2_, gmO3_, gmOutR_, gmOutI_;
    TBuf<TPosition::VECCALC> bufO0_, bufO1_, bufO2_, bufO3_;
    TPipe pipe_;
};

extern "C" __global__ __aicore__ void ifft_postproc(
    GM_ADDR o0, GM_ADDR o1, GM_ADDR o2, GM_ADDR o3,
    GM_ADDR outR, GM_ADDR outI, GM_ADDR tiling)
{
    IfftPostprocKernel op;
    op.Init(o0, o1, o2, o3, outR, outI, tiling);
    op.Process();
}
