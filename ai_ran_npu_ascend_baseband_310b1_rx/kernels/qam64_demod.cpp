/**
 * @file qam64_demod.cpp — 64-QAM hard demodulation
 *
 * Quantizes equalized symbols to nearest constellation point and extracts bits.
 * Natural binary mapping: idx 0-7 → bits [b2, b1, b0] = binary(idx).
 *
 * Input:  qamR/I [1192, 224] half (padded from 220 data subcarriers)
 * Output: bits   [1192, 1344] int8 — 6 bits per symbol: [b2I, b1I, b0I, b2Q, b1Q, b0Q]
 *         First 1320 columns valid (220 × 6), last 24 columns zero-padded.
 *
 * Pipeline: CopyIn (DMA) → Compute (Vector quantize + scalar bit extract) → CopyOut (DMA)
 * Quantization: all-vector (Cast → Muls → Adds → Clamp → Cast)
 * Bit extraction: 8-unrolled scalar loop (NPU scalar unit bottleneck)
 *
 * GM args (3): qamR, qamI, bitsOut
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t USE_CORES    = 8;
constexpr int32_t M_TOTAL      = 1192;
constexpr int32_t N_IN_PAD     = 224;    // 16-aligned input width
constexpr int32_t N_DATA       = 220;    // actual data subcarriers
constexpr int32_t BITS_PER_SYM = 6;
constexpr int32_t N_OUT        = 1344;   // 224 × 6, 32-byte aligned
constexpr int32_t N_DATA_BITS  = N_DATA * BITS_PER_SYM;  // 1320

constexpr float SCALE_F  = 3.24037f;    // sqrt(42) / 2
constexpr float OFFSET_F = 3.5f;

class Qam64DemodKernel {
public:
    __aicore__ inline Qam64DemodKernel() {}

    __aicore__ inline void Init(GM_ADDR qamR, GM_ADDR qamI, GM_ADDR bitsOut, TPipe *pipe) {
        this->pipe = pipe;
        int32_t blockIdx = GetBlockIdx();
        int32_t perCore = M_TOTAL / USE_CORES, rem = M_TOTAL % USE_CORES;
        if (blockIdx < rem) {
            myStartRow = blockIdx * (perCore + 1); myRows = perCore + 1;
        } else {
            myStartRow = rem * (perCore + 1) + (blockIdx - rem) * perCore; myRows = perCore;
        }

        inRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qamR), M_TOTAL * N_IN_PAD);
        inIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(qamI), M_TOTAL * N_IN_PAD);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(bitsOut), M_TOTAL * N_OUT);

        pipe->InitBuffer(inQR, 2, N_IN_PAD * (int32_t)sizeof(half));
        pipe->InitBuffer(inQI, 2, N_IN_PAD * (int32_t)sizeof(half));
        pipe->InitBuffer(fBufR, N_IN_PAD * (int32_t)sizeof(float));
        pipe->InitBuffer(fBufI, N_IN_PAD * (int32_t)sizeof(float));
        pipe->InitBuffer(idxBufR, N_IN_PAD * (int32_t)sizeof(int32_t));
        pipe->InitBuffer(idxBufI, N_IN_PAD * (int32_t)sizeof(int32_t));
        pipe->InitBuffer(outQ, 2, N_OUT * (int32_t)sizeof(int8_t));
    }

    __aicore__ inline void Process() {
        for (int32_t r = 0; r < myRows; r++) {
            CopyIn(r);
            Compute(r);
            CopyOut(r);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t r) {
        int32_t row = myStartRow + r;
        auto rLocal = inQR.AllocTensor<half>();
        auto iLocal = inQI.AllocTensor<half>();
        DataCopy(rLocal, inRGm[row * N_IN_PAD], N_IN_PAD);
        DataCopy(iLocal, inIGm[row * N_IN_PAD], N_IN_PAD);
        inQR.EnQue(rLocal);
        inQI.EnQue(iLocal);
    }

    __aicore__ inline void Compute(int32_t r) {
        auto rIn = inQR.DeQue<half>();
        auto iIn = inQI.DeQue<half>();
        auto oLocal = outQ.AllocTensor<int8_t>();

        LocalTensor<float> fR = fBufR.Get<float>(N_IN_PAD);
        LocalTensor<float> fI = fBufI.Get<float>(N_IN_PAD);
        LocalTensor<int32_t> idxR = idxBufR.Get<int32_t>(N_IN_PAD);
        LocalTensor<int32_t> idxI = idxBufI.Get<int32_t>(N_IN_PAD);

        // Vector quantization: half → float → scale+offset → round → clamp [0,7]
        Cast(fR, rIn, RoundMode::CAST_NONE, N_IN_PAD);
        Muls(fR, fR, SCALE_F, N_IN_PAD);
        Adds(fR, fR, OFFSET_F, N_IN_PAD);
        Cast(idxR, fR, RoundMode::CAST_RINT, N_IN_PAD);
        Maxs(idxR, idxR, (int32_t)0, N_IN_PAD);
        Mins(idxR, idxR, (int32_t)7, N_IN_PAD);

        Cast(fI, iIn, RoundMode::CAST_NONE, N_IN_PAD);
        Muls(fI, fI, SCALE_F, N_IN_PAD);
        Adds(fI, fI, OFFSET_F, N_IN_PAD);
        Cast(idxI, fI, RoundMode::CAST_RINT, N_IN_PAD);
        Maxs(idxI, idxI, (int32_t)0, N_IN_PAD);
        Mins(idxI, idxI, (int32_t)7, N_IN_PAD);

        PipeBarrier<PIPE_V>();

        // Scalar bit extraction (8-unrolled)
        constexpr int32_t FULL_ITERS = N_DATA / 8;
        for (int32_t blk = 0; blk < FULL_ITERS; blk++) {
            int32_t base = blk * 8;
            int32_t i0 = idxR.GetValue(base),   i1 = idxR.GetValue(base + 1);
            int32_t i2 = idxR.GetValue(base + 2), i3 = idxR.GetValue(base + 3);
            int32_t i4 = idxR.GetValue(base + 4), i5 = idxR.GetValue(base + 5);
            int32_t i6 = idxR.GetValue(base + 6), i7 = idxR.GetValue(base + 7);
            int32_t q0 = idxI.GetValue(base),   q1 = idxI.GetValue(base + 1);
            int32_t q2 = idxI.GetValue(base + 2), q3 = idxI.GetValue(base + 3);
            int32_t q4 = idxI.GetValue(base + 4), q5 = idxI.GetValue(base + 5);
            int32_t q6 = idxI.GetValue(base + 6), q7 = idxI.GetValue(base + 7);

            int32_t o = base * BITS_PER_SYM;
            #define EMIT_BITS(off, iv, qv) \
                oLocal.SetValue(off+0,(int8_t)((iv>>2)&1)); \
                oLocal.SetValue(off+1,(int8_t)((iv>>1)&1)); \
                oLocal.SetValue(off+2,(int8_t)(iv&1));       \
                oLocal.SetValue(off+3,(int8_t)((qv>>2)&1)); \
                oLocal.SetValue(off+4,(int8_t)((qv>>1)&1)); \
                oLocal.SetValue(off+5,(int8_t)(qv&1))

            EMIT_BITS(o,      i0, q0); EMIT_BITS(o + 6,  i1, q1);
            EMIT_BITS(o + 12, i2, q2); EMIT_BITS(o + 18, i3, q3);
            EMIT_BITS(o + 24, i4, q4); EMIT_BITS(o + 30, i5, q5);
            EMIT_BITS(o + 36, i6, q6); EMIT_BITS(o + 42, i7, q7);
            #undef EMIT_BITS
        }
        // Remainder (220 % 8 = 4 symbols)
        for (int32_t s = FULL_ITERS * 8; s < N_DATA; s++) {
            int32_t iv = idxR.GetValue(s), qv = idxI.GetValue(s);
            int32_t o = s * BITS_PER_SYM;
            oLocal.SetValue(o + 0, (int8_t)((iv >> 2) & 1));
            oLocal.SetValue(o + 1, (int8_t)((iv >> 1) & 1));
            oLocal.SetValue(o + 2, (int8_t)(iv & 1));
            oLocal.SetValue(o + 3, (int8_t)((qv >> 2) & 1));
            oLocal.SetValue(o + 4, (int8_t)((qv >> 1) & 1));
            oLocal.SetValue(o + 5, (int8_t)(qv & 1));
        }
        // Zero padding
        for (int32_t i = N_DATA_BITS; i < N_OUT; i++)
            oLocal.SetValue(i, (int8_t)0);

        inQR.FreeTensor(rIn);
        inQI.FreeTensor(iIn);
        outQ.EnQue(oLocal);
    }

    __aicore__ inline void CopyOut(int32_t r) {
        auto oLocal = outQ.DeQue<int8_t>();
        DataCopy(outGm[(myStartRow + r) * N_OUT], oLocal, N_OUT);
        outQ.FreeTensor(oLocal);
    }

private:
    TPipe *pipe;
    int32_t myStartRow, myRows;
    GlobalTensor<half> inRGm, inIGm;
    GlobalTensor<int8_t> outGm;
    TQue<TPosition::VECIN, 2> inQR, inQI;
    TQue<TPosition::VECOUT, 2> outQ;
    TBuf<TPosition::VECCALC> fBufR, fBufI, idxBufR, idxBufI;
};

extern "C" __global__ __aicore__ void qam64_demod(
    GM_ADDR qamR, GM_ADDR qamI, GM_ADDR bitsOut)
{
    TPipe pipe;
    Qam64DemodKernel kernel;
    kernel.Init(qamR, qamI, bitsOut, &pipe);
    kernel.Process();
}
