/**
 * @file fine_sync.cpp — Frame synchronization (3-phase, 8-core parallel)
 *
 * Phase 1: Energy scan → coarse_pos   (8-core parallel, each scans 1/8 range)
 * Phase 2: Cross-correlation → fine_pos (8-core parallel search)
 * Phase 3: CP-based CFO estimation     (8-core parallel, each handles nSym/8)
 *
 * 3-launch protocol (controlled by pybind):
 *   Launch 1: searchStart=0  → Phase 1 only, write coarsePos per core
 *   Launch 2: searchStart>0  → Phase 2 only, 8-core parallel xcorr
 *   Launch 3: searchStart=-1 → Phase 3 only, 8-core parallel CFO
 *
 * GM args (8): rawR, rawI, refR, refI, outR, outI, peakOut, tiling
 * Output per core: [pos, power, P_real, P_imag, compLen, 0, 0, 0]
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include "kernel_operator.h"

constexpr int32_t SYNC_CORES   = 8;
constexpr int32_t REF_L        = 1024;
constexpr int32_t CHUNK        = 4096;
constexpr int32_t PEAK_STRIDE  = 8;
constexpr int32_t E_BLOCK      = 256;
constexpr int32_t E_SCAN_TILE  = 2048;

struct FineSyncTiling {
    int32_t searchStart; int32_t searchLen; int32_t refLen; int32_t totalN;
    int32_t nSym;        int32_t symUp;     int32_t nFftUp; int32_t cpUp;
    int32_t nSymDecode;  int32_t zcUp;
};

__aicore__ inline void CopyTiling(FineSyncTiling &t, GM_ADDR gm) {
    auto s = reinterpret_cast<__gm__ uint32_t*>(gm);
    auto d = reinterpret_cast<uint32_t*>(&t);
    for (uint32_t i = 0; i < sizeof(FineSyncTiling) / sizeof(uint32_t); i++)
        d[i] = s[i];
}

class FineSyncKernel {
private:
    AscendC::TPipe *pipePtr;
    FineSyncTiling til;
    int32_t coreIdx, refLen;
    AscendC::GlobalTensor<half>  rawRGm, rawIGm, refRGm, refIGm;
    AscendC::GlobalTensor<float> peakOutGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inRBuf, inIBuf, sigRBuf, sigIBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> refRBuf, refIBuf, tmpABuf, tmpBBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> peakBuf;

public:
    __aicore__ inline FineSyncKernel() {}

    __aicore__ inline void Init(GM_ADDR rawR, GM_ADDR rawI, GM_ADDR refR, GM_ADDR refI,
        GM_ADDR outR, GM_ADDR outI, GM_ADDR peakOut, GM_ADDR tilingGm, AscendC::TPipe *pipe)
    {
        pipePtr = pipe;
        CopyTiling(til, tilingGm);
        coreIdx = (int32_t)AscendC::GetBlockIdx();
        refLen = til.refLen;
        if (refLen > REF_L) refLen = REF_L;

        rawRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rawR), til.totalN);
        rawIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rawI), til.totalN);
        refRGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(refR), refLen);
        refIGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(refI), refLen);
        peakOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(peakOut) + coreIdx * PEAK_STRIDE, PEAK_STRIDE);

        pipe->InitBuffer(inRBuf,  CHUNK * sizeof(half));
        pipe->InitBuffer(inIBuf,  CHUNK * sizeof(half));
        pipe->InitBuffer(sigRBuf, CHUNK * sizeof(float));
        pipe->InitBuffer(sigIBuf, CHUNK * sizeof(float));
        pipe->InitBuffer(refRBuf, REF_L * sizeof(float));
        pipe->InitBuffer(refIBuf, REF_L * sizeof(float));
        pipe->InitBuffer(tmpABuf, CHUNK * sizeof(float));
        pipe->InitBuffer(tmpBBuf, CHUNK * sizeof(float));
        pipe->InitBuffer(peakBuf, PEAK_STRIDE * sizeof(float));
    }

    __aicore__ inline float TreeReduceSum(AscendC::LocalTensor<float> &buf, int32_t len) {
        int32_t n = len;
        while (n >= 16) {
            int32_t h = n / 2;
            AscendC::Add(buf, buf, buf[h], h);
            AscendC::PipeBarrier<PIPE_V>();
            n = h;
        }
        float s = 0.0f;
        for (int32_t i = 0; i < n; i++) s += buf.GetValue(i);
        return s;
    }

    // ========== Phase 1: Energy-based coarse sync (8-core parallel) ==========

    static constexpr int32_t LOW_RUN      = 4;
    static constexpr float   HIGH_RATIO   = 4.0f;
    static constexpr float   MIN_CONTRAST = 4.0f;
    static constexpr int32_t CONFIRM_WIN  = 3;
    static constexpr float   CONFIRM_RATIO = 2.0f;

    __aicore__ inline float ComputeBlockEnergy(int32_t pos,
        AscendC::LocalTensor<half> &inR, AscendC::LocalTensor<half> &inI,
        AscendC::LocalTensor<float> &sigR, AscendC::LocalTensor<float> &sigI,
        AscendC::LocalTensor<float> &tmpA)
    {
        int32_t al = E_BLOCK;
        if (pos + al > til.totalN) { al = (til.totalN - pos) / 16 * 16; if (al <= 0) return 0.0f; }
        AscendC::DataCopy(inR, rawRGm[pos], al);
        AscendC::DataCopy(inI, rawIGm[pos], al);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(sigR, inR, AscendC::RoundMode::CAST_NONE, al);
        AscendC::Cast(sigI, inI, AscendC::RoundMode::CAST_NONE, al);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmpA, sigR, sigR, al);
        AscendC::Mul(sigR, sigI, sigI, al);
        AscendC::Add(tmpA, tmpA, sigR, al);
        AscendC::PipeBarrier<PIPE_V>();
        return TreeReduceSum(tmpA, al);
    }

    __aicore__ inline int32_t Phase1_EnergyCoarse_Parallel(int32_t maxValidPos) {
        int32_t totalW = (til.totalN - E_BLOCK) / E_BLOCK;
        if (totalW <= 0) return -1;

        auto inR  = inRBuf.Get<half>(E_SCAN_TILE);
        auto inI  = inIBuf.Get<half>(E_SCAN_TILE);
        auto sigR = sigRBuf.Get<float>(E_SCAN_TILE);
        auto sigI = sigIBuf.Get<float>(E_SCAN_TILE);
        auto tmpA = tmpABuf.Get<float>(E_SCAN_TILE);

        // Global max-energy pre-scan (stride=64, all cores)
        float maxE = 0.0f;
        for (int32_t w = 0; w < totalW; w += 64) {
            float e = ComputeBlockEnergy(w * E_BLOCK, inR, inI, sigR, sigI, tmpA);
            if (e > maxE) maxE = e;
        }
        if (maxE <= 0) return -1;

        // Per-core range split
        int32_t perCore = totalW / SYNC_CORES;
        int32_t rem = totalW % SYNC_CORES;
        int32_t myWStart, myWCount;
        if (coreIdx < rem) {
            myWStart = coreIdx * (perCore + 1);
            myWCount = perCore + 1;
        } else {
            myWStart = rem * (perCore + 1) + (coreIdx - rem) * perCore;
            myWCount = perCore;
        }

        // Warm-up: build running average from blocks before this core's range
        float runBuf[4];
        int32_t runCount = 0;
        float runSum = 0.0f;
        int32_t warmStart = myWStart - LOW_RUN;
        if (warmStart < 0) warmStart = 0;
        for (int32_t w = warmStart; w < myWStart; w++) {
            int32_t pos = w * E_BLOCK;
            if (pos + E_BLOCK > til.totalN) break;
            float e = ComputeBlockEnergy(pos, inR, inI, sigR, sigI, tmpA);
            if (runCount < LOW_RUN) {
                runBuf[runCount] = e; runSum += e; runCount++;
            } else {
                int32_t idx = w % LOW_RUN;
                runSum -= runBuf[idx]; runSum += e; runBuf[idx] = e;
            }
        }

        // Main scan over this core's range
        for (int32_t i = 0; i < myWCount; i++) {
            int32_t w = myWStart + i;
            int32_t pos = w * E_BLOCK;
            if (pos + E_BLOCK > til.totalN) break;
            if (pos > maxValidPos) break;

            float e = ComputeBlockEnergy(pos, inR, inI, sigR, sigI, tmpA);

            if (runCount >= LOW_RUN) {
                float runAvg = runSum / (float)LOW_RUN;
                if (e > runAvg * HIGH_RATIO && runAvg < maxE / MIN_CONTRAST) {
                    // Confirm: read blocks beyond core range (shared GM, safe)
                    bool confirmed = true;
                    for (int32_t c = 1; c <= CONFIRM_WIN; c++) {
                        int32_t cpos = (w + c) * E_BLOCK;
                        if (cpos + E_BLOCK > til.totalN) { confirmed = false; break; }
                        float ce = ComputeBlockEnergy(cpos, inR, inI, sigR, sigI, tmpA);
                        if (ce < runAvg * CONFIRM_RATIO) { confirmed = false; break; }
                    }
                    if (confirmed) return pos;
                }
            }

            if (runCount < LOW_RUN) {
                runBuf[runCount] = e; runSum += e; runCount++;
            } else {
                int32_t idx = w % LOW_RUN;
                runSum -= runBuf[idx]; runSum += e; runBuf[idx] = e;
            }
        }
        return -1;
    }

    // ========== Phase 2: Fine sync via normalized cross-correlation ==========

    static constexpr float XCORR_NORM_THRESH = 0.1f;

    __aicore__ inline void Phase2_FineSync(int32_t searchStart, int32_t searchLen,
                                           int32_t &bestPos, float &bestPower)
    {
        bestPos = -1; bestPower = 0.0f;
        if (searchLen <= 0) return;

        auto inR  = inRBuf.Get<half>(CHUNK);
        auto inI  = inIBuf.Get<half>(CHUNK);
        auto sigR = sigRBuf.Get<float>(CHUNK);
        auto sigI = sigIBuf.Get<float>(CHUNK);
        auto rR   = refRBuf.Get<float>(REF_L);
        auto rI   = refIBuf.Get<float>(REF_L);
        auto tmpA = tmpABuf.Get<float>(CHUNK);
        auto tmpB = tmpBBuf.Get<float>(CHUNK);

        // Load and cast reference signal (once)
        AscendC::DataCopy(inR, refRGm, refLen);
        AscendC::DataCopy(inI, refIGm, refLen);
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t refAl = (refLen / 16) * 16;
        if (refAl == 0) refAl = 16;
        AscendC::Cast(rR, inR, AscendC::RoundMode::CAST_NONE, refAl);
        AscendC::Cast(rI, inI, AscendC::RoundMode::CAST_NONE, refAl);
        AscendC::PipeBarrier<PIPE_V>();

        // Reference energy (once)
        AscendC::Mul(tmpA, rR, rR, refAl);
        AscendC::Mul(tmpB, rI, rI, refAl);
        AscendC::Add(tmpA, tmpA, tmpB, refAl);
        AscendC::PipeBarrier<PIPE_V>();
        float refEnergy = TreeReduceSum(tmpA, refAl);
        if (refEnergy < 1e-20f) refEnergy = 1e-20f;

        for (int32_t n = 0; n < searchLen; n++) {
            int32_t pos = searchStart + n;
            if (pos + refLen > til.totalN) break;

            AscendC::DataCopy(inR, rawRGm[pos], refAl);
            AscendC::DataCopy(inI, rawIGm[pos], refAl);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Cast once, reuse for xcorr real/imag and energy
            AscendC::Cast(sigR, inR, AscendC::RoundMode::CAST_NONE, refAl);
            AscendC::Cast(sigI, inI, AscendC::RoundMode::CAST_NONE, refAl);
            AscendC::PipeBarrier<PIPE_V>();

            // Cross-correlation real: sum(sigR*rR + sigI*rI)
            AscendC::Mul(tmpA, sigR, rR, refAl);
            AscendC::Mul(tmpB, sigI, rI, refAl);
            AscendC::Add(tmpA, tmpA, tmpB, refAl);
            AscendC::PipeBarrier<PIPE_V>();
            float xr = TreeReduceSum(tmpA, refAl);

            // Cross-correlation imag: sum(sigI*rR - sigR*rI)
            AscendC::Mul(tmpA, sigI, rR, refAl);
            AscendC::Mul(tmpB, sigR, rI, refAl);
            AscendC::Sub(tmpA, tmpA, tmpB, refAl);
            AscendC::PipeBarrier<PIPE_V>();
            float xi = TreeReduceSum(tmpA, refAl);

            float xcorrPower = xr * xr + xi * xi;

            // Signal energy (reuse sigR/sigI — TreeReduce only modifies tmpA/tmpB)
            AscendC::Mul(tmpA, sigR, sigR, refAl);
            AscendC::Mul(tmpB, sigI, sigI, refAl);
            AscendC::Add(tmpA, tmpA, tmpB, refAl);
            AscendC::PipeBarrier<PIPE_V>();
            float sigEnergy = TreeReduceSum(tmpA, refAl);
            if (sigEnergy < 1e-20f) sigEnergy = 1e-20f;

            // Normalized filter: reject false peaks (normPow < 0.01 = noise)
            float normPow = xcorrPower / (sigEnergy * refEnergy);
            if (normPow < XCORR_NORM_THRESH) continue;

            if (xcorrPower > bestPower) {
                bestPower = xcorrPower;
                bestPos = pos;
            }
        }
    }

    // ========== Phase 3: CP-based CFO estimation (8-core parallel) ==========

    __aicore__ inline void Phase3_CpCfo_Parallel(int32_t finePos, float &Pr, float &Pi) {
        Pr = 0; Pi = 0;
        if (finePos < 0) return;

        int32_t fStart = finePos + refLen;
        int32_t nSym = til.nSym;
        if (fStart + nSym * til.symUp > til.totalN)
            nSym = (til.totalN - fStart) / til.symUp;
        if (nSym <= 0) return;

        // Split symbols across cores
        int32_t perCore = nSym / SYNC_CORES;
        int32_t rem = nSym % SYNC_CORES;
        int32_t mySymStart, mySymCount;
        if (coreIdx < rem) {
            mySymStart = coreIdx * (perCore + 1); mySymCount = perCore + 1;
        } else {
            mySymStart = rem * (perCore + 1) + (coreIdx - rem) * perCore; mySymCount = perCore;
        }

        int32_t cpA = (til.cpUp / 16) * 16;
        if (cpA == 0) cpA = 16;

        auto inR   = inRBuf.Get<half>(CHUNK);
        auto inI   = inIBuf.Get<half>(CHUNK);
        auto cpRf  = sigRBuf.Get<float>(CHUNK);
        auto tailRf = sigIBuf.Get<float>(CHUNK);
        auto cpIf  = refRBuf.Get<float>(REF_L);
        auto tailIf = refIBuf.Get<float>(REF_L);
        auto tmpA  = tmpABuf.Get<float>(CHUNK);
        auto tmpB  = tmpBBuf.Get<float>(CHUNK);

        for (int32_t s = 0; s < mySymCount; s++) {
            int32_t symS = fStart + (mySymStart + s) * til.symUp;
            int32_t tailP = symS + til.nFftUp;
            if (tailP + cpA > til.totalN) break;

            AscendC::DataCopy(inR, rawRGm[symS], cpA);
            AscendC::DataCopy(inR[cpA], rawRGm[tailP], cpA);
            AscendC::DataCopy(inI, rawIGm[symS], cpA);
            AscendC::DataCopy(inI[cpA], rawIGm[tailP], cpA);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Cast(cpRf, inR, AscendC::RoundMode::CAST_NONE, cpA);
            AscendC::Cast(tailRf, inR[cpA], AscendC::RoundMode::CAST_NONE, cpA);
            AscendC::Cast(cpIf, inI, AscendC::RoundMode::CAST_NONE, cpA);
            AscendC::Cast(tailIf, inI[cpA], AscendC::RoundMode::CAST_NONE, cpA);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(tmpA, cpRf, tailRf, cpA);
            AscendC::Mul(tmpB, cpIf, tailIf, cpA);
            AscendC::Add(tmpA, tmpA, tmpB, cpA);
            AscendC::PipeBarrier<PIPE_V>();
            Pr += TreeReduceSum(tmpA, cpA);

            AscendC::Mul(tmpA, cpIf, tailRf, cpA);
            AscendC::Mul(tmpB, cpRf, tailIf, cpA);
            AscendC::Sub(tmpA, tmpA, tmpB, cpA);
            AscendC::PipeBarrier<PIPE_V>();
            Pi += TreeReduceSum(tmpA, cpA);
        }
    }

    // ========== Utilities ==========

    __aicore__ inline void WriteEmpty() {
        auto peak = peakBuf.Get<float>(PEAK_STRIDE);
        for (int32_t i = 0; i < PEAK_STRIDE; i++) peak.SetValue(i, 0.0f);
        peak.SetValue(0, -1.0f);
        AscendC::DataCopy(peakOutGm, peak, PEAK_STRIDE);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // ========== Main dispatch ==========

    __aicore__ inline void Process() {
        auto peak = peakBuf.Get<float>(PEAK_STRIDE);

        int32_t frameTotal = refLen + til.nSymDecode * til.symUp;
        int32_t maxValidPos = til.totalN - frameTotal;
        if (maxValidPos < 0) maxValidPos = 0;

        // Phase 3 mode: searchStart < 0
        if (til.searchStart < 0) {
            int32_t finePos = til.searchLen;
            float Pr = 0, Pi = 0;
            Phase3_CpCfo_Parallel(finePos, Pr, Pi);
            peak.SetValue(0, (float)finePos);
            peak.SetValue(1, 0.0f);
            peak.SetValue(2, Pr);
            peak.SetValue(3, Pi);
            for (int32_t i = 4; i < PEAK_STRIDE; i++) peak.SetValue(i, 0.0f);
            AscendC::DataCopy(peakOutGm, peak, PEAK_STRIDE);
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        // Phase 1 mode: searchStart == 0
        if (til.searchStart == 0) {
            int32_t coarsePos = Phase1_EnergyCoarse_Parallel(maxValidPos);
            peak.SetValue(0, (float)coarsePos);
            peak.SetValue(1, (coarsePos >= 0) ? 1.0f : 0.0f);
            for (int32_t i = 2; i < PEAK_STRIDE; i++) peak.SetValue(i, 0.0f);
            AscendC::DataCopy(peakOutGm, peak, PEAK_STRIDE);
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        // Phase 2 mode: searchStart > 0 (8-core parallel xcorr)
        {
            int32_t ss = til.searchStart, sl = til.searchLen;
            if (ss + sl > maxValidPos) sl = maxValidPos - ss;
            if (sl < 0) sl = 0;

            int32_t perCore = sl / SYNC_CORES, rem2 = sl % SYNC_CORES;
            int32_t coarsePos, searchLen;
            if (coreIdx < rem2) {
                coarsePos = ss + coreIdx * (perCore + 1); searchLen = perCore + 1;
            } else {
                coarsePos = ss + rem2 * (perCore + 1) + (coreIdx - rem2) * perCore; searchLen = perCore;
            }
            if (coarsePos + searchLen > maxValidPos) searchLen = maxValidPos - coarsePos;
            if (searchLen < 0) searchLen = 0;

            int32_t finePos; float finePower;
            Phase2_FineSync(coarsePos, searchLen, finePos, finePower);
            if (finePos < 0 || finePower == 0.0f) { WriteEmpty(); return; }
            if (finePos > maxValidPos) { WriteEmpty(); return; }

            peak.SetValue(0, (float)finePos);
            peak.SetValue(1, finePower);
            for (int32_t i = 2; i < PEAK_STRIDE; i++) peak.SetValue(i, 0.0f);
            AscendC::DataCopy(peakOutGm, peak, PEAK_STRIDE);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
};

extern "C" __global__ __aicore__ void fine_sync(
    GM_ADDR rawR, GM_ADDR rawI, GM_ADDR refR, GM_ADDR refI,
    GM_ADDR outR, GM_ADDR outI, GM_ADDR peakOut, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    FineSyncKernel kernel;
    kernel.Init(rawR, rawI, refR, refI, outR, outI, peakOut, tilingGm, &pipe);
    kernel.Process();
}
