/**
 * @file ls_equalizer.cpp — LS channel estimation + ZF equalization
 *
 * Two-stage processing per tile of rows:
 *   Stage 1 (Matmul): H_est = pilots × LS_matrix  [tile, 32] × [32, 512] → [tile, 512]
 *   Stage 2 (Vector): eq = rx / H_est (ZF equalization with regularization)
 *
 * LS_matrix is precomputed: pseudo-inverse mapping pilot observations to
 * full channel estimates (256 real + 256 imag = 512 outputs).
 *
 * GM args (8): pilots, lsMatrix, rxReal, rxImag, eqReal, eqImag, workspace, tilingGm
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

constexpr int32_t USE_CORES     = 8;
constexpr int32_t SUBCARRIERS   = 256;
constexpr int32_t LS_OUTPUT_DIM = 512;
constexpr int32_t PILOT_DIM     = 32;
constexpr int32_t TILE_ROWS     = 32;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize,
                                   int32_t &totalRows, GM_ADDR tilingGM)
{
    auto ptr = reinterpret_cast<uint32_t*>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t*>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++)
        ptr[i] = src[i];
    uint32_t offset = sizeof(TCubeTiling);
    localMemSize = *reinterpret_cast<__gm__ uint64_t*>(tilingGM + offset);
    totalRows = *reinterpret_cast<__gm__ int32_t*>(tilingGM + offset + sizeof(uint64_t));
}

class LSEqualizerKernel {
public:
    __aicore__ inline LSEqualizerKernel() {}

    __aicore__ inline void Init(GM_ADDR pilots, GM_ADDR lsMatrix,
                               GM_ADDR rxReal, GM_ADDR rxImag,
                               GM_ADDR eqReal, GM_ADDR eqImag,
                               GM_ADDR workspace,
                               int32_t totalRows,
                               const TCubeTiling &tiling, AscendC::TPipe *pipe)
    {
        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t rowsPerCore = totalRows / USE_CORES;
        int32_t remainder = totalRows % USE_CORES;
        if (blockIdx < remainder) {
            myStartRow = blockIdx * (rowsPerCore + 1); myRows = rowsPerCore + 1;
        } else {
            myStartRow = remainder * (rowsPerCore + 1) + (blockIdx - remainder) * rowsPerCore;
            myRows = rowsPerCore;
        }

        pilotsGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(pilots) + myStartRow * PILOT_DIM,
                                 myRows * PILOT_DIM);
        lsMatrixGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(lsMatrix), PILOT_DIM * LS_OUTPUT_DIM);
        rxRealGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rxReal) + myStartRow * SUBCARRIERS,
                                 myRows * SUBCARRIERS);
        rxImagGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rxImag) + myStartRow * SUBCARRIERS,
                                 myRows * SUBCARRIERS);
        eqRealGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(eqReal) + myStartRow * SUBCARRIERS,
                                 myRows * SUBCARRIERS);
        eqImagGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(eqImag) + myStartRow * SUBCARRIERS,
                                 myRows * SUBCARRIERS);

        pipe->InitBuffer(mmWorkBuf, 32 * 1024);
        pipe->InitBuffer(lsChannelBuf, TILE_ROWS * LS_OUTPUT_DIM * sizeof(float));
        pipe->InitBuffer(inQueueRxR, 2, SUBCARRIERS * sizeof(half));
        pipe->InitBuffer(inQueueRxI, 2, SUBCARRIERS * sizeof(half));
        pipe->InitBuffer(outQueueEqR, 2, SUBCARRIERS * sizeof(half));
        pipe->InitBuffer(outQueueEqI, 2, SUBCARRIERS * sizeof(half));
        pipe->InitBuffer(zfWorkBuf, SUBCARRIERS * 6 * sizeof(half));

        this->pipe = pipe;
        this->tiling = tiling;
        this->wsPtr = workspace;
    }

    __aicore__ inline void Process() {
        int32_t processedRows = 0;
        while (processedRows < myRows) {
            int32_t tileRows = (myRows - processedRows >= TILE_ROWS)
                               ? TILE_ROWS : (myRows - processedRows);
            PerformLSEstimation(processedRows, tileRows);

            AscendC::TQueSync<PIPE_M, PIPE_V> sync;
            sync.SetFlag(0); sync.WaitFlag(0);

            for (int32_t row = 0; row < tileRows; row++) {
                CopyInSignal(processedRows + row);
                PerformZFEqualization(row);
                CopyOutResult(processedRows + row);
            }
            processedRows += tileRows;
        }
    }

private:
    __aicore__ inline void PerformLSEstimation(int32_t rowOffset, int32_t tileRows) {
        AscendC::LocalTensor<float> lsResult = lsChannelBuf.Get<float>(TILE_ROWS * LS_OUTPUT_DIM);

        Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
               MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
               MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>> mm;
        REGIST_MATMUL_OBJ(pipe, wsPtr, mm, &tiling);

        AscendC::LocalTensor<uint8_t> mmWork = mmWorkBuf.Get<uint8_t>(32 * 1024);
        mm.SetLocalWorkspace(mmWork);
        mm.SetOrgShape(tileRows, LS_OUTPUT_DIM, PILOT_DIM, PILOT_DIM);
        mm.SetTensorA(pilotsGm[rowOffset * PILOT_DIM], false);
        mm.SetTensorB(lsMatrixGm, false);
        mm.SetTail(tileRows, LS_OUTPUT_DIM);
        mm.IterateAll(lsResult);
        mm.End();
    }

    __aicore__ inline void CopyInSignal(int32_t rowIdx) {
        auto rxR = inQueueRxR.AllocTensor<half>();
        auto rxI = inQueueRxI.AllocTensor<half>();
        AscendC::DataCopy(rxR, rxRealGm[rowIdx * SUBCARRIERS], SUBCARRIERS);
        AscendC::DataCopy(rxI, rxImagGm[rowIdx * SUBCARRIERS], SUBCARRIERS);
        inQueueRxR.EnQue(rxR);
        inQueueRxI.EnQue(rxI);
    }

    __aicore__ inline void PerformZFEqualization(int32_t tileRowIdx) {
        auto rxR = inQueueRxR.DeQue<half>();
        auto rxI = inQueueRxI.DeQue<half>();
        auto eqR = outQueueEqR.AllocTensor<half>();
        auto eqI = outQueueEqI.AllocTensor<half>();
        auto zfWork = zfWorkBuf.Get<half>(SUBCARRIERS * 6);

        auto hReal = zfWork;
        auto hImag = zfWork[SUBCARRIERS];
        auto h2    = zfWork[SUBCARRIERS * 2];
        auto temp1 = zfWork[SUBCARRIERS * 3];
        auto temp2 = zfWork[SUBCARRIERS * 4];
        auto recip = zfWork[SUBCARRIERS * 5];

        // Extract channel estimate from Matmul result
        auto lsResult = lsChannelBuf.Get<float>(TILE_ROWS * LS_OUTPUT_DIM);
        int32_t lsOff = tileRowIdx * LS_OUTPUT_DIM;
        AscendC::Cast(hReal, lsResult[lsOff], AscendC::RoundMode::CAST_NONE, SUBCARRIERS);
        AscendC::Cast(hImag, lsResult[lsOff + SUBCARRIERS], AscendC::RoundMode::CAST_NONE, SUBCARRIERS);

        // ZF equalization: eq = rx * conj(H) / (|H|^2 + eps)
        AscendC::Mul(h2, hReal, hReal, SUBCARRIERS);
        AscendC::Mul(temp1, hImag, hImag, SUBCARRIERS);
        AscendC::Add(h2, h2, temp1, SUBCARRIERS);
        AscendC::Adds(h2, h2, static_cast<half>(1e-6), SUBCARRIERS);
        AscendC::Reciprocal(recip, h2, SUBCARRIERS);

        AscendC::Mul(temp1, rxR, hReal, SUBCARRIERS);
        AscendC::Mul(temp2, rxI, hImag, SUBCARRIERS);
        AscendC::Add(temp1, temp1, temp2, SUBCARRIERS);
        AscendC::Mul(eqR, temp1, recip, SUBCARRIERS);

        AscendC::Mul(temp1, rxI, hReal, SUBCARRIERS);
        AscendC::Mul(temp2, rxR, hImag, SUBCARRIERS);
        AscendC::Sub(temp1, temp1, temp2, SUBCARRIERS);
        AscendC::Mul(eqI, temp1, recip, SUBCARRIERS);

        outQueueEqR.EnQue(eqR); outQueueEqI.EnQue(eqI);
        inQueueRxR.FreeTensor(rxR); inQueueRxI.FreeTensor(rxI);
    }

    __aicore__ inline void CopyOutResult(int32_t rowIdx) {
        auto eqR = outQueueEqR.DeQue<half>();
        auto eqI = outQueueEqI.DeQue<half>();
        AscendC::DataCopy(eqRealGm[rowIdx * SUBCARRIERS], eqR, SUBCARRIERS);
        AscendC::DataCopy(eqImagGm[rowIdx * SUBCARRIERS], eqI, SUBCARRIERS);
        outQueueEqR.FreeTensor(eqR); outQueueEqI.FreeTensor(eqI);
    }

private:
    AscendC::TPipe *pipe;
    TCubeTiling tiling;
    GM_ADDR wsPtr;
    int32_t myStartRow, myRows;
    AscendC::GlobalTensor<half> pilotsGm, lsMatrixGm;
    AscendC::GlobalTensor<half> rxRealGm, rxImagGm, eqRealGm, eqImagGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lsChannelBuf, zfWorkBuf, mmWorkBuf;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueRxR, inQueueRxI;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueEqR, outQueueEqI;
};

extern "C" __global__ __aicore__ void ls_equalizer(
    GM_ADDR pilots, GM_ADDR lsMatrix, GM_ADDR rxReal, GM_ADDR rxImag,
    GM_ADDR eqReal, GM_ADDR eqImag, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
    int32_t totalRows = 0;
    CopyTiling(&tiling, localMemSize, totalRows, tilingGm);

    LSEqualizerKernel kernel;
    kernel.Init(pilots, lsMatrix, rxReal, rxImag, eqReal, eqImag,
                workspace, totalRows, tiling, &pipe);
    kernel.Process();
}
