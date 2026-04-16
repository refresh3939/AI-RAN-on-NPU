/**
 * @file data_extract_mm.cpp — Data subcarrier extraction via Matmul permutation
 *
 * Replaces scalar-loop extraction with a Matmul: data = eq × P
 *   eq:   [1192, 256] half (equalized frequency-domain signal)
 *   P:    [256, 224]  half (permutation matrix: P[data_sc[k], k] = 1.0)
 *   data: [1192, 224] half (extracted data subcarriers)
 *
 * Two passes: real and imaginary processed separately.
 * ~5-10x faster than the scalar extract_subcarriers kernel.
 *
 * GM args (7): inR, inI, outR, outI, permMatrix, workspace, tilingGm
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
constexpr int32_t K_DIM     = 256;    // N_FFT: input columns
constexpr int32_t N_DIM     = 224;    // K_DATA_PAD: output columns
constexpr int32_t TILE_M    = 32;

class DataExtractMmKernel {
public:
    __aicore__ inline DataExtractMmKernel() {}

    __aicore__ inline void Init(
        GM_ADDR inR, GM_ADDR inI, GM_ADDR outR, GM_ADDR outI,
        GM_ADDR permMatrix, GM_ADDR workspace,
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

        inRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(inR), M_TOTAL * K_DIM);
        inIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(inI), M_TOTAL * K_DIM);
        outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outR), M_TOTAL * N_DIM);
        outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outI), M_TOTAL * N_DIM);
        permGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(permMatrix), K_DIM * N_DIM);

        pipe->InitBuffer(mmWorkBuf, 32 * 1024);
        pipe->InitBuffer(mmResultBuf, TILE_M * N_DIM * (int32_t)sizeof(float));
        pipe->InitBuffer(outQ, 2, TILE_M * N_DIM * (int32_t)sizeof(half));

        REGIST_MATMUL_OBJ(pipe, workspace, mm, &(this->cubeTiling));
        AscendC::LocalTensor<uint8_t> mmWork = mmWorkBuf.Get<uint8_t>(32 * 1024);
        mm.SetLocalWorkspace(mmWork);
    }

    __aicore__ inline void Process() {
        if (myRows <= 0) return;
        DoPass(inRGm, outRGm);
        DoPass(inIGm, outIGm);
    }

private:
    __aicore__ inline void DoPass(
        AscendC::GlobalTensor<half> &inGm, AscendC::GlobalTensor<half> &outGm)
    {
        AscendC::LocalTensor<float> mmOut = mmResultBuf.Get<float>(TILE_M * N_DIM);
        int32_t processed = 0;

        while (processed < myRows) {
            int32_t tile = (myRows - processed >= TILE_M) ? TILE_M : (myRows - processed);

            mm.SetOrgShape(tile, N_DIM, K_DIM);
            mm.SetTensorA(inGm[(myStartRow + processed) * K_DIM], false);
            mm.SetTensorB(permGm, false);
            mm.SetTail(tile, N_DIM);
            mm.IterateAll(mmOut);
            mm.End();

            AscendC::TQueSync<PIPE_M, PIPE_V> sync;
            sync.SetFlag(0); sync.WaitFlag(0);

            for (int32_t r = 0; r < tile; r++) {
                auto oLocal = outQ.AllocTensor<half>();
                AscendC::Cast(oLocal, mmOut[r * N_DIM], AscendC::RoundMode::CAST_NONE, N_DIM);
                outQ.EnQue(oLocal);
                oLocal = outQ.DeQue<half>();
                AscendC::DataCopy(outGm[(myStartRow + processed + r) * N_DIM], oLocal, N_DIM);
                outQ.FreeTensor(oLocal);
            }
            processed += tile;
        }
    }

private:
    AscendC::TPipe *pipe;
    TCubeTiling cubeTiling;
    int32_t myStartRow, myRows;
    AscendC::GlobalTensor<half> inRGm, inIGm, outRGm, outIGm, permGm;
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>> mm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmWorkBuf, mmResultBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQ;
};

extern "C" __global__ __aicore__ void data_extract_mm(
    GM_ADDR inR, GM_ADDR inI, GM_ADDR outR, GM_ADDR outI,
    GM_ADDR permMatrix, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling cubeTiling;
    auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
    auto ptr = reinterpret_cast<uint32_t*>(&cubeTiling);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++) ptr[i] = src[i];

    DataExtractMmKernel kernel;
    kernel.Init(inR, inI, outR, outI, permMatrix, workspace, cubeTiling, &pipe);
    kernel.Process();
}
