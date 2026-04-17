/**
 * @file up_sample_rrc.cpp — RRC polyphase upsampling (Matmul Toeplitz)
 *
 * Performs polyphase Root-Raised-Cosine filtering and 4× upsampling as four
 * independent Matmul operations (one per phase):
 *   Y_p = X · H_p^T,    [totalRows, 272] × [272, 272]^T → [totalRows, 272]
 * for p = 0, 1, 2, 3. Called twice per frame (real and imag channels).
 *
 * Dynamic row count: totalRows is read from tilingGm at offset=512.
 *   - Used by TX chain with totalRows ≈ 1268 (overlap-add layout)
 *
 * Output layout: phase-major
 *   outR[p * totalRows * SYM_IN : (p+1) * totalRows * SYM_IN] = phase p real
 *
 * GM args (7): inReal, inImag, filterCoeff, outReal, outImag, workspace, tiling
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

constexpr int32_t USE_CORES = 8;
constexpr int32_t SYM_IN    = 272;
constexpr int32_t SPS       = 4;
constexpr int32_t MAX_ROWS  = 1300;        // upper bound for buffer sizing
constexpr int32_t TILE_M    = 32;
constexpr int32_t TOTAL_ROWS_OFFSET = 512; // byte offset in tilingGm

class UpSampleRrcKernel {
public:
    __aicore__ inline UpSampleRrcKernel() {}

    __aicore__ inline void Init(
        GM_ADDR inReal, GM_ADDR inImag, GM_ADDR filterCoeff,
        GM_ADDR outReal, GM_ADDR outImag,
        const TCubeTiling &cubeTiling, int32_t totalRows,
        AscendC::TPipe *pipe)
    {
        this->pipe = pipe;
        this->cubeTiling = cubeTiling;
        this->totalRows = totalRows;

        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t perCore = totalRows / USE_CORES;
        int32_t rem = totalRows % USE_CORES;
        if (blockIdx < rem) {
            myStartRow = blockIdx * (perCore + 1);
            myRows = perCore + 1;
        } else {
            myStartRow = rem * (perCore + 1) + (blockIdx - rem) * perCore;
            myRows = perCore;
        }

        inRGm.SetGlobalBuffer((__gm__ half*)inReal, MAX_ROWS * SYM_IN);
        inIGm.SetGlobalBuffer((__gm__ half*)inImag, MAX_ROWS * SYM_IN);
        int32_t outSize = SPS * MAX_ROWS * SYM_IN;
        outRGm.SetGlobalBuffer((__gm__ half*)outReal, outSize);
        outIGm.SetGlobalBuffer((__gm__ half*)outImag, outSize);

        auto fcHalf = reinterpret_cast<__gm__ half*>(filterCoeff);
        for (int p = 0; p < SPS; p++)
            hGm[p].SetGlobalBuffer(fcHalf + p * SYM_IN * SYM_IN, SYM_IN * SYM_IN);

        pipe->InitBuffer(mmWorkBuf,   32 * 1024);
        pipe->InitBuffer(mmResultBuf, TILE_M * SYM_IN * sizeof(float));
        pipe->InitBuffer(outQ, 2,     TILE_M * SYM_IN * sizeof(half));

        REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mm, &(this->cubeTiling));
        mm.SetLocalWorkspace(mmWorkBuf.Get<uint8_t>(32 * 1024));
    }

    __aicore__ inline void Process() {
        for (int32_t p = 0; p < SPS; p++) {
            int32_t phaseOff = p * totalRows * SYM_IN;
            DoConvPhase(inRGm, hGm[p], outRGm, phaseOff);
            DoConvPhase(inIGm, hGm[p], outIGm, phaseOff);
        }
    }

private:
    __aicore__ inline void DoConvPhase(
        AscendC::GlobalTensor<half> &inGm,
        AscendC::GlobalTensor<half> &filterGm,
        AscendC::GlobalTensor<half> &outGm,
        int32_t phaseOff)
    {
        auto mmOut = mmResultBuf.Get<float>(TILE_M * SYM_IN);
        int32_t processed = 0;

        while (processed < myRows) {
            int32_t tile = (myRows - processed >= TILE_M) ? TILE_M
                                                           : (myRows - processed);
            int32_t rowOff = (myStartRow + processed) * SYM_IN;

            mm.SetOrgShape(tile, SYM_IN, SYM_IN, SYM_IN);
            mm.SetTensorA(inGm[rowOff], false);
            mm.SetTensorB(filterGm, true);
            mm.SetTail(tile, SYM_IN);
            mm.IterateAll(mmOut);
            mm.End();

            AscendC::TQueSync<PIPE_M, PIPE_V> sync;
            sync.SetFlag(0); sync.WaitFlag(0);

            for (int32_t r = 0; r < tile; r++) {
                auto outLocal = outQ.AllocTensor<half>();
                AscendC::Cast(outLocal, mmOut[r * SYM_IN],
                              AscendC::RoundMode::CAST_NONE, SYM_IN);
                outQ.EnQue(outLocal);
                outLocal = outQ.DeQue<half>();
                int32_t gmOff = phaseOff + (myStartRow + processed + r) * SYM_IN;
                AscendC::DataCopy(outGm[gmOff], outLocal, SYM_IN);
                outQ.FreeTensor(outLocal);
            }
            processed += tile;
        }
    }

    AscendC::TPipe *pipe;
    TCubeTiling cubeTiling;
    int32_t totalRows;
    int32_t myStartRow, myRows;
    AscendC::GlobalTensor<half> inRGm, inIGm, outRGm, outIGm;
    AscendC::GlobalTensor<half> hGm[4];
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>> mm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmWorkBuf, mmResultBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
};

extern "C" __global__ __aicore__ void up_sample_rrc(
    GM_ADDR inReal, GM_ADDR inImag, GM_ADDR filterCoeff,
    GM_ADDR outReal, GM_ADDR outImag,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;

    TCubeTiling cubeTiling;
    {
        uint32_t *ptr = reinterpret_cast<uint32_t*>(&cubeTiling);
        auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
        for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++)
            *ptr = *(src + i);
    }

    auto tilingBytes = reinterpret_cast<__gm__ uint8_t*>(tilingGm);
    int32_t totalRows = *reinterpret_cast<__gm__ int32_t*>(tilingBytes + TOTAL_ROWS_OFFSET);

    UpSampleRrcKernel kernel;
    kernel.Init(inReal, inImag, filterCoeff, outReal, outImag,
                cubeTiling, totalRows, &pipe);
    kernel.Process();
}
