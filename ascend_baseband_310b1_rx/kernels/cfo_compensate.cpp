/**
 * @file cfo_compensate.cpp — Time-domain CFO compensation (vector cos/sin)
 *
 * Generates compensation phasor entirely on-chip using angle-sum identities:
 *   base[i] = cos/sin(delta*i)         — computed once via scalar recursion
 *   comp[i] = base[i] * tile_rotation  — 6 vector ops per tile
 *
 * Tiling layout (88 bytes, precomputed by pybind):
 *   [0]     int32: totalN
 *   [1..2]  float: cos_delta, sin_delta
 *   [3..4]  float: tile_rot_cos, tile_rot_sin
 *   [5]     reserved
 *   [6..21] float: 8 cores × (cos_init, sin_init)
 *
 * GM args (5): sigR, sigI, outR, outI, tilingGm
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"

constexpr int32_t COMP_CORES = 8;
constexpr int32_t COMP_TILE  = 4096;
constexpr int32_t ALIGN      = 16;

constexpr int32_t T_TOTALN       = 0;
constexpr int32_t T_COS_DELTA    = 1;
constexpr int32_t T_SIN_DELTA    = 2;
constexpr int32_t T_TILE_ROT_COS = 3;
constexpr int32_t T_TILE_ROT_SIN = 4;
constexpr int32_t T_CORE_BASE    = 6;

class CfoCompensateKernel {
public:
    __aicore__ inline CfoCompensateKernel() {}

    __aicore__ inline void Init(
        GM_ADDR sigR, GM_ADDR sigI, GM_ADDR outR, GM_ADDR outI,
        GM_ADDR tilingGm, AscendC::TPipe *pipe)
    {
        this->pipe = pipe;
        int32_t blockIdx = AscendC::GetBlockIdx();

        auto tg = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
        this->totalN = (int32_t)tg[T_TOTALN];
        uint32_t u;
        u = tg[T_COS_DELTA];    cos_delta = *reinterpret_cast<float*>(&u);
        u = tg[T_SIN_DELTA];    sin_delta = *reinterpret_cast<float*>(&u);
        u = tg[T_TILE_ROT_COS]; tile_rot_cos = *reinterpret_cast<float*>(&u);
        u = tg[T_TILE_ROT_SIN]; tile_rot_sin = *reinterpret_cast<float*>(&u);

        int32_t co = T_CORE_BASE + blockIdx * 2;
        u = tg[co];     tile_cos = *reinterpret_cast<float*>(&u);
        u = tg[co + 1]; tile_sin = *reinterpret_cast<float*>(&u);

        // 8-core workload split
        int32_t totalTiles = (totalN + COMP_TILE - 1) / COMP_TILE;
        int32_t tilesPerCore = totalTiles / COMP_CORES;
        int32_t remTiles = totalTiles % COMP_CORES;
        int32_t myTileStart, myTileCount;
        if (blockIdx < remTiles) {
            myTileStart = blockIdx * (tilesPerCore + 1);
            myTileCount = tilesPerCore + 1;
        } else {
            myTileStart = remTiles * (tilesPerCore + 1) +
                          (blockIdx - remTiles) * tilesPerCore;
            myTileCount = tilesPerCore;
        }
        myStart = myTileStart * COMP_TILE;
        myLen = myTileCount * COMP_TILE;
        if (myStart + myLen > totalN) myLen = totalN - myStart;
        if (myStart >= totalN) myLen = 0;

        sigRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(sigR), totalN);
        sigIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(sigI), totalN);
        outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outR), totalN);
        outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(outI), totalN);

        pipe->InitBuffer(baseCosBuf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(baseSinBuf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(sRBuf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(sIBuf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(cRBuf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(cIBuf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(t1Buf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(t2Buf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(oRBuf, COMP_TILE * sizeof(half));
        pipe->InitBuffer(oIBuf, COMP_TILE * sizeof(half));
    }

    __aicore__ inline void ComputeBase(
        AscendC::LocalTensor<half> &bCos, AscendC::LocalTensor<half> &bSin)
    {
        float cd = cos_delta, sd = sin_delta;
        float c = 1.0f, s = 0.0f;
        for (int32_t i = 0; i < COMP_TILE; i++) {
            bCos.SetValue(i, (half)c);
            bSin.SetValue(i, (half)s);
            float nc = c * cd - s * sd;
            float ns = s * cd + c * sd;
            c = nc; s = ns;
        }
    }

    __aicore__ inline void Process() {
        if (myLen <= 0) return;

        auto bCos = baseCosBuf.Get<half>(COMP_TILE);
        auto bSin = baseSinBuf.Get<half>(COMP_TILE);
        auto sR = sRBuf.Get<half>(COMP_TILE);
        auto sI = sIBuf.Get<half>(COMP_TILE);
        auto cR = cRBuf.Get<half>(COMP_TILE);
        auto cI = cIBuf.Get<half>(COMP_TILE);
        auto t1 = t1Buf.Get<half>(COMP_TILE);
        auto t2 = t2Buf.Get<half>(COMP_TILE);
        auto oR = oRBuf.Get<half>(COMP_TILE);
        auto oI = oIBuf.Get<half>(COMP_TILE);

        ComputeBase(bCos, bSin);

        float tc = tile_cos, ts = tile_sin;
        float trc = tile_rot_cos, trs = tile_rot_sin;

        int32_t processed = 0;
        while (processed < myLen) {
            int32_t remain = myLen - processed;
            int32_t tile = (remain >= COMP_TILE) ? COMP_TILE : remain;
            int32_t alignedTile = (tile / ALIGN) * ALIGN;
            if (alignedTile == 0) {
                float nc = tc * trc - ts * trs;
                float ns = ts * trc + tc * trs;
                tc = nc; ts = ns;
                processed += tile;
                continue;
            }

            int32_t gmPos = myStart + processed;

            // Generate compensation phasor via angle-sum identity
            half tc_h = (half)tc, ts_h = (half)ts;
            AscendC::Duplicate(t1, tc_h, alignedTile);
            AscendC::Duplicate(t2, ts_h, alignedTile);
            AscendC::Mul(cR, bCos, t1, alignedTile);
            AscendC::Mul(oR, bSin, t2, alignedTile);
            AscendC::Sub(cR, cR, oR, alignedTile);
            AscendC::Mul(cI, bSin, t1, alignedTile);
            AscendC::Mul(oI, bCos, t2, alignedTile);
            AscendC::Add(cI, cI, oI, alignedTile);
            AscendC::PipeBarrier<PIPE_V>();

            // Load signal
            AscendC::DataCopy(sR, sigRGm[gmPos], alignedTile);
            AscendC::DataCopy(sI, sigIGm[gmPos], alignedTile);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Complex multiply: out = sig * comp
            AscendC::Mul(oR, sR, cR, alignedTile);
            AscendC::Mul(t1, sI, cI, alignedTile);
            AscendC::Sub(oR, oR, t1, alignedTile);
            AscendC::Mul(oI, sR, cI, alignedTile);
            AscendC::Mul(t1, sI, cR, alignedTile);
            AscendC::Add(oI, oI, t1, alignedTile);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopy(outRGm[gmPos], oR, alignedTile);
            AscendC::DataCopy(outIGm[gmPos], oI, alignedTile);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Advance tile rotation
            float nc = tc * trc - ts * trs;
            float ns = ts * trc + tc * trs;
            tc = nc; ts = ns;

            processed += tile;
        }
    }

private:
    AscendC::TPipe *pipe;
    int32_t totalN, myStart, myLen;
    float cos_delta, sin_delta;
    float tile_cos, tile_sin;
    float tile_rot_cos, tile_rot_sin;
    AscendC::GlobalTensor<half> sigRGm, sigIGm, outRGm, outIGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> baseCosBuf, baseSinBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sRBuf, sIBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cRBuf, cIBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> t1Buf, t2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oRBuf, oIBuf;
};

extern "C" __global__ __aicore__ void cfo_compensate(
    GM_ADDR sigR, GM_ADDR sigI, GM_ADDR outR, GM_ADDR outI, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    CfoCompensateKernel kernel;
    kernel.Init(sigR, sigI, outR, outI, tilingGm, &pipe);
    kernel.Process();
}
