/**
 * @file qam64_modulation.cpp — 64-QAM modulation with IFFT-aligned output
 *
 * Maps packed 6-bit QAM bytes to complex baseband I/Q levels:
 *   byte (0..63) → I_level = (byte>>3) × SCALE + OFFSET
 *                  Q_level = (byte&7) × SCALE + OFFSET
 *   SCALE=2/√42, OFFSET=-7/√42 (unit average power)
 *
 * Output stride is K_IN=224 (32B-aligned for IFFT input layout), with the
 * trailing 4 columns per row padded to zero. This eliminates the host-side
 * stride-pad step after QAM modulation.
 *
 * Input:  [N_SYM * K_DATA] uint8, one packed 6-bit symbol per byte (MSB-first)
 * Output: [N_SYM, K_IN] half R/I
 *
 * Each core processes N_SYM/USE_CORES rows; 1 row per tile.
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t N_SYM     = 1192;
constexpr int32_t K_DATA    = 220;
constexpr int32_t K_IN      = 224;                    // output stride (32B aligned)
constexpr int32_t USE_CORES = 8;
constexpr int32_t ROWS_PER_CORE = N_SYM / USE_CORES;  // 149

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_PAD   = K_IN;                  // 224

constexpr float SCALE  = 2.0f * 0.15430335f;
constexpr float OFFSET = -7.0f * 0.15430335f;

class Qam64ModulationKernel {
public:
    __aicore__ inline Qam64ModulationKernel() {}

    __aicore__ inline void Init(GM_ADDR inputBits, GM_ADDR outputReal,
                                GM_ADDR outputImag, TPipe *pipeIn)
    {
        int32_t blockIdx = GetBlockIdx();
        int32_t startRow = blockIdx * ROWS_PER_CORE;
        rowCount_ = ROWS_PER_CORE;

        inputBitsGm_.SetGlobalBuffer(
            (__gm__ uint8_t*)inputBits + startRow * K_DATA, rowCount_ * K_DATA);
        outputRealGm_.SetGlobalBuffer(
            (__gm__ half*)outputReal + startRow * K_IN, rowCount_ * K_IN);
        outputImagGm_.SetGlobalBuffer(
            (__gm__ half*)outputImag + startRow * K_IN, rowCount_ * K_IN);

        pipe_ = pipeIn;
        pipe_->InitBuffer(inQueueBits_,  BUFFER_NUM, TILE_PAD * sizeof(uint8_t));
        pipe_->InitBuffer(outQueueReal_, BUFFER_NUM, TILE_PAD * sizeof(half));
        pipe_->InitBuffer(outQueueImag_, BUFFER_NUM, TILE_PAD * sizeof(half));
        pipe_->InitBuffer(workBufH1_,    TILE_PAD * sizeof(half));
        pipe_->InitBuffer(workBufH2_,    TILE_PAD * sizeof(half));
        pipe_->InitBuffer(workBufH3_,    TILE_PAD * sizeof(half));
        pipe_->InitBuffer(workBufInt_,   TILE_PAD * sizeof(int16_t));
    }

    __aicore__ inline void Process() {
        for (int32_t r = 0; r < rowCount_; r++) {
            CopyIn(r);
            Compute(r);
            CopyOut(r);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t row) {
        auto bits = inQueueBits_.AllocTensor<uint8_t>();
        DataCopy(bits, inputBitsGm_[row * K_DATA], TILE_PAD);
        inQueueBits_.EnQue(bits);
    }

    __aicore__ inline void Compute(int32_t row) {
        auto bits = inQueueBits_.DeQue<uint8_t>();
        auto outR = outQueueReal_.AllocTensor<half>();
        auto outI = outQueueImag_.AllocTensor<half>();

        auto byteH   = workBufH1_.Get<half>();
        auto idxI    = workBufH2_.Get<half>();
        auto idxQ    = workBufH3_.Get<half>();
        auto tempInt = workBufInt_.Get<int16_t>();

        // uint8 → half
        Cast(byteH, bits, RoundMode::CAST_NONE, TILE_PAD);

        // I_idx = floor(byte / 8)
        Muls(idxI, byteH, (half)0.125f, TILE_PAD);
        Cast(tempInt, idxI, RoundMode::CAST_FLOOR, TILE_PAD);
        Cast(idxI, tempInt, RoundMode::CAST_NONE, TILE_PAD);

        // Q_idx = byte - I_idx × 8
        Muls(idxQ, idxI, (half)8.0f, TILE_PAD);
        Sub(idxQ, byteH, idxQ, TILE_PAD);

        // level = idx × SCALE + OFFSET
        Muls(idxI, idxI, (half)SCALE, TILE_PAD);
        Adds(idxI, idxI, (half)OFFSET, TILE_PAD);
        Muls(idxQ, idxQ, (half)SCALE, TILE_PAD);
        Adds(idxQ, idxQ, (half)OFFSET, TILE_PAD);

        DataCopy(outR, idxI, TILE_PAD);
        DataCopy(outI, idxQ, TILE_PAD);

        // Zero-pad trailing columns [K_DATA..K_IN) for 32B alignment
        for (int32_t i = K_DATA; i < TILE_PAD; i++) {
            outR.SetValue(i, (half)0.0f);
            outI.SetValue(i, (half)0.0f);
        }

        outQueueReal_.EnQue(outR);
        outQueueImag_.EnQue(outI);
        inQueueBits_.FreeTensor(bits);
    }

    __aicore__ inline void CopyOut(int32_t row) {
        auto outR = outQueueReal_.DeQue<half>();
        auto outI = outQueueImag_.DeQue<half>();
        DataCopy(outputRealGm_[row * K_IN], outR, TILE_PAD);
        DataCopy(outputImagGm_[row * K_IN], outI, TILE_PAD);
        outQueueReal_.FreeTensor(outR);
        outQueueImag_.FreeTensor(outI);
    }

    TPipe *pipe_;
    TQue<TPosition::VECIN,  BUFFER_NUM> inQueueBits_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueReal_, outQueueImag_;
    GlobalTensor<uint8_t> inputBitsGm_;
    GlobalTensor<half>    outputRealGm_, outputImagGm_;
    TBuf<TPosition::VECCALC> workBufH1_, workBufH2_, workBufH3_, workBufInt_;
    int32_t rowCount_;
};

extern "C" __global__ __aicore__ void qam64_modulation(
    GM_ADDR input_bits, GM_ADDR output_real, GM_ADDR output_imag)
{
    TPipe pipe;
    Qam64ModulationKernel op;
    op.Init(input_bits, output_real, output_imag, &pipe);
    op.Process();
}
