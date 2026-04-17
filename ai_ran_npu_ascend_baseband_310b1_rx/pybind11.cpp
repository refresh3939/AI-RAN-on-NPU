/**
 * @file pybind11.cpp — Ascend baseband processing platform (pybind interface)
 *
 * Exposes the complete OFDM RX chain to Python via pybind11:
 *   init_resources() — allocate GM buffers, upload constants
 *   run_h2d()        — sc16 → f16 conversion (ARM NEON) + DMA upload
 *   run_sync()       — 3-phase frame synchronization (8-core parallel)
 *   run_decode()     — CFO → RRC → FFT → EQ → QAM → LDPC pipeline
 *
 * Optimizations:
 *   - 3-launch parallel sync (Phase1 energy + Phase2 xcorr + Phase3 CFO)
 *   - RRC outputs directly into FFT stride-272 layout (zero-copy CP removal)
 *   - Data extraction via Matmul permutation matrix
 *   - LDPC async batching (no workspace memset, single final sync)
 *   - Zero intermediate syncs in decode pipeline (stream ordering)
 *   - ARM NEON sc16→f16 with pre-allocated buffers
 *
 * Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
 * Target: Ascend 310B1 (Orange Pi AI Pro)
 * SPDX-License-Identifier: MIT
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <arm_neon.h>

#include "acl/acl.h"
#include "data_utils.h"

#include "aclrtlaunch_fine_sync.h"
#include "aclrtlaunch_cfo_compensate.h"
#include "aclrtlaunch_rrc_downsample.h"
#include "aclrtlaunch_ofdm_fft.h"
#include "aclrtlaunch_fft_postproc.h"
#include "aclrtlaunch_extract_subcarriers.h"
#include "aclrtlaunch_ls_equalizer.h"
#include "aclrtlaunch_qam64_demod.h"
#include "aclrtlaunch_ldpc_decode.h"
#include "aclrtlaunch_data_extract_mm.h"

extern "C" void RrcGenerateTiling(const char *socVersion, uint8_t *tilingBuf);
extern "C" void FftGenerateTiling(const char *socVersion, uint8_t *tilingBuf);
extern "C" void LdpcGenerateTiling(const char *socVersion, uint8_t *tilingBuf);
extern "C" void LsGenerateTiling(const char *socVersion, uint8_t *tilingBuf);
extern "C" void DataExtGenerateTiling(const char *socVersion, uint8_t *tilingBuf);

namespace py = pybind11;

// ============================================================
// System constants
// ============================================================
static constexpr int32_t BLOCK_DIM   = 8;
static constexpr int32_t N_FFT       = 256;
static constexpr int32_t NCP         = 16;
static constexpr int32_t SPS         = 4;
static constexpr int32_t N_SYM       = 1192;
static constexpr int32_t SYM_TD      = N_FFT + NCP;         // 272
static constexpr int32_t SYM_UP      = SYM_TD * SPS;        // 1088
static constexpr int32_t ZC_N        = 256;
static constexpr int32_t ZC_UP       = ZC_N * SPS;           // 1024
static constexpr int32_t K_DATA      = 220;
static constexpr int32_t K_DATA_PAD  = 224;                  // 32B-aligned
static constexpr int32_t K_PILOT     = 16;
static constexpr int32_t PILOT_DIM   = 32;                   // K_PILOT * 2 (R/I interleaved)
static constexpr int32_t CAPTURE_LEN = 2800000;
static constexpr int32_t FRAME_LEN   = N_SYM * SYM_UP;
static constexpr int32_t COMP_LEN    = ZC_UP + FRAME_LEN;

static constexpr int32_t LDPC_N_CODEWORD = 512;
static constexpr int32_t LDPC_BATCH      = 256;
static constexpr int32_t LDPC_TOTAL_BITS = N_SYM * K_DATA * 6;
static constexpr int32_t LDPC_TOTAL_CW   = LDPC_TOTAL_BITS / LDPC_N_CODEWORD;
static constexpr int32_t LDPC_N_BATCHES  = (LDPC_TOTAL_CW + LDPC_BATCH - 1) / LDPC_BATCH;
static constexpr int32_t LDPC_ALLOC_CW   = LDPC_N_BATCHES * LDPC_BATCH;
static constexpr int32_t QAM_OUT_COLS    = 1344;             // K_DATA_PAD * 6

static constexpr int32_t CFO_TILING_WORDS = 22;
static constexpr int32_t CFO_COMP_TILE    = 4096;

// ============================================================
// Tiling structures
// ============================================================
struct FineSyncTiling {
    int32_t searchStart, searchLen, refLen, totalN;
    int32_t nSym, symUp, nFftUp, cpUp;
    int32_t nSymDecode, zcUp;
};
struct FftPostTiling  { int32_t totalRows; int32_t nFFT; };
struct ExtractScTiling { int32_t totalRows; int32_t nFFT; int32_t K; int32_t mode; int32_t padK; };

// ============================================================
// Global state
// ============================================================
static aclrtStream g_stream = nullptr;
static bool g_initialized = false;
static uint16_t *g_r16 = nullptr, *g_i16 = nullptr;  // pre-allocated H2D buffers

// GM buffer pointers (allocated in init_resources)
static void *gm_rx_r, *gm_rx_i;
static void *gm_sync_refR, *gm_sync_refI, *gm_sync_outR, *gm_sync_outI;
static void *gm_sync_peak, *gm_sync_tiling;
static void *gm_comp_r, *gm_comp_i, *gm_cfo_tiling;
static void *gm_rrc_r, *gm_rrc_i, *gm_rrc_coeff, *gm_rrc_ws, *gm_rrc_tiling;
static void *gm_fft_out0, *gm_fft_out1, *gm_fft_out2, *gm_fft_out3;
static void *gm_fft_matB_cos, *gm_fft_matB_sin, *gm_fft_ws, *gm_fft_tiling;
static void *gm_freq_r, *gm_freq_i, *gm_dc_r, *gm_dc_i, *gm_fftpost_tiling;
static void *gm_pilot, *gm_pilot_sc, *gm_extract_pilot_tiling;
static void *gm_ls_matrix, *gm_eq_r, *gm_eq_i, *gm_ls_ws, *gm_ls_tiling;
static void *gm_data_r, *gm_data_i, *gm_data_sc, *gm_extract_data_tiling;
static void *gm_perm_matrix, *gm_dataext_ws, *gm_dataext_tiling;
static void *gm_qam_out;
static void *gm_ldpc_a, *gm_ldpc_b, *gm_ldpc_c1, *gm_ldpc_mask;
static void *gm_ldpc_c, *gm_ldpc_ginv, *gm_ldpc_info, *gm_ldpc_ws, *gm_ldpc_tiling;
static void *gm_dummy;

// ============================================================
// Helpers
// ============================================================
using clk = std::chrono::high_resolution_clock;
static inline double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void *AllocGM(size_t bytes) {
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(ptr, bytes, 0, bytes));
    return ptr;
}
static void UploadGM(void *dev, const void *host, size_t bytes) {
    CHECK_ACL(aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
static void DownloadGM(void *host, const void *dev, size_t bytes) {
    CHECK_ACL(aclrtMemcpy(host, bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
}

static void PrepareCfoTiling(uint32_t *buf, int32_t totalN, double cfoHz, double fs,
                             int32_t startOffset = 0) {
    double delta = -2.0 * M_PI * cfoHz / fs;
    float cd = (float)cos(delta), sd = (float)sin(delta);
    double tile_angle = delta * CFO_COMP_TILE;
    float trc = (float)cos(tile_angle), trs = (float)sin(tile_angle);
    buf[0] = (uint32_t)totalN;
    memcpy(&buf[1], &cd, 4); memcpy(&buf[2], &sd, 4);
    memcpy(&buf[3], &trc, 4); memcpy(&buf[4], &trs, 4);
    buf[5] = 0;
    int32_t totalTiles = (totalN + CFO_COMP_TILE - 1) / CFO_COMP_TILE;
    for (int32_t core = 0; core < 8; core++) {
        int32_t tpc = totalTiles / 8, rem = totalTiles % 8;
        int32_t myStart = (core < rem) ? core * (tpc + 1) : rem * (tpc + 1) + (core - rem) * tpc;
        double phase = delta * (double)(startOffset + myStart * CFO_COMP_TILE);
        float c0 = (float)cos(phase), s0 = (float)sin(phase);
        memcpy(&buf[6 + core * 2], &c0, 4);
        memcpy(&buf[7 + core * 2], &s0, 4);
    }
}

// ============================================================
// init_resources — allocate all GM buffers and upload constants
// ============================================================
static void init_resources(
    py::array_t<uint16_t> dc_r, py::array_t<uint16_t> dc_i,
    py::array_t<int32_t> pilot_sc, py::array_t<int32_t> data_sc,
    py::array_t<uint16_t> ref_r, py::array_t<uint16_t> ref_i,
    py::array_t<uint16_t> rrc_coeff,
    py::array_t<uint16_t> fft_cos, py::array_t<uint16_t> fft_sin,
    py::array_t<uint16_t> ls_matrix,
    py::array_t<int8_t> ldpc_H, py::array_t<int8_t> ldpc_ginv)
{
    if (g_initialized) return;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateStream(&g_stream));

    // Allocate GM buffers
    gm_rx_r = AllocGM(CAPTURE_LEN * 2);        gm_rx_i = AllocGM(CAPTURE_LEN * 2);
    gm_sync_refR = AllocGM(ZC_UP * 2);         gm_sync_refI = AllocGM(ZC_UP * 2);
    gm_sync_outR = AllocGM(64);                 gm_sync_outI = AllocGM(64);
    gm_sync_peak = AllocGM(64 * 4);             gm_sync_tiling = AllocGM(256);
    gm_comp_r = AllocGM(COMP_LEN * 2);          gm_comp_i = AllocGM(COMP_LEN * 2);
    gm_cfo_tiling = AllocGM(256);
    gm_rrc_r = AllocGM(N_SYM * SYM_TD * 2);    gm_rrc_i = AllocGM(N_SYM * SYM_TD * 2);
    gm_rrc_coeff = AllocGM(rrc_coeff.size()*2); gm_rrc_ws = AllocGM(4*1024*1024);
    gm_rrc_tiling = AllocGM(1024);
    gm_fft_out0 = AllocGM(N_SYM*N_FFT*2);      gm_fft_out1 = AllocGM(N_SYM*N_FFT*2);
    gm_fft_out2 = AllocGM(N_SYM*N_FFT*2);      gm_fft_out3 = AllocGM(N_SYM*N_FFT*2);
    gm_fft_matB_cos = AllocGM(fft_cos.size()*2); gm_fft_matB_sin = AllocGM(fft_sin.size()*2);
    gm_fft_ws = AllocGM(4*1024*1024);            gm_fft_tiling = AllocGM(1024);
    gm_freq_r = AllocGM(N_SYM*N_FFT*2);         gm_freq_i = AllocGM(N_SYM*N_FFT*2);
    gm_dc_r = AllocGM(N_FFT*2);                 gm_dc_i = AllocGM(N_FFT*2);
    gm_fftpost_tiling = AllocGM(64);
    gm_pilot = AllocGM(N_SYM*PILOT_DIM*2);      gm_pilot_sc = AllocGM(K_PILOT*4);
    gm_extract_pilot_tiling = AllocGM(64);
    gm_ls_matrix = AllocGM(ls_matrix.size()*2);
    gm_eq_r = AllocGM(N_SYM*N_FFT*2);           gm_eq_i = AllocGM(N_SYM*N_FFT*2);
    gm_ls_ws = AllocGM(4*1024*1024);             gm_ls_tiling = AllocGM(1024);
    gm_data_r = AllocGM(N_SYM*K_DATA_PAD*2);    gm_data_i = AllocGM(N_SYM*K_DATA_PAD*2);
    gm_data_sc = AllocGM(K_DATA_PAD*4);          gm_extract_data_tiling = AllocGM(64);
    gm_perm_matrix = AllocGM(N_FFT*K_DATA_PAD*2);
    gm_dataext_ws = AllocGM(4*1024*1024);        gm_dataext_tiling = AllocGM(1024);
    gm_qam_out = AllocGM(N_SYM*QAM_OUT_COLS);
    gm_ldpc_a = AllocGM(LDPC_ALLOC_CW*LDPC_N_CODEWORD);
    gm_ldpc_b = AllocGM(LDPC_N_CODEWORD*256);   gm_ldpc_c1 = AllocGM(LDPC_BATCH*256);
    gm_ldpc_mask = AllocGM(64*4);
    gm_ldpc_c = AllocGM(LDPC_BATCH*LDPC_N_CODEWORD*4);
    gm_ldpc_ginv = AllocGM(256*256);
    gm_ldpc_info = AllocGM(LDPC_ALLOC_CW*256*2); gm_ldpc_ws = AllocGM(4*1024*1024);
    gm_ldpc_tiling = AllocGM(8192);
    gm_dummy = AllocGM(64);

    // Upload constants
    UploadGM(gm_sync_refR, ref_r.data(), ref_r.size()*2);
    UploadGM(gm_sync_refI, ref_i.data(), ref_i.size()*2);
    UploadGM(gm_rrc_coeff, rrc_coeff.data(), rrc_coeff.size()*2);
    UploadGM(gm_fft_matB_cos, fft_cos.data(), fft_cos.size()*2);
    UploadGM(gm_fft_matB_sin, fft_sin.data(), fft_sin.size()*2);
    UploadGM(gm_dc_r, dc_r.data(), dc_r.size()*2);
    UploadGM(gm_dc_i, dc_i.data(), dc_i.size()*2);
    UploadGM(gm_pilot_sc, pilot_sc.data(), pilot_sc.size()*4);
    UploadGM(gm_data_sc, data_sc.data(), data_sc.size()*4);
    UploadGM(gm_ls_matrix, ls_matrix.data(), ls_matrix.size()*2);
    UploadGM(gm_ldpc_b, ldpc_H.data(), ldpc_H.size());
    UploadGM(gm_ldpc_ginv, ldpc_ginv.data(), ldpc_ginv.size());

    // Generate and upload tiling data
    { const char *soc = "Ascend310B1"; uint8_t tb[8192];
      memset(tb,0,sizeof(tb)); RrcGenerateTiling(soc,tb);     UploadGM(gm_rrc_tiling,tb,1024);
      memset(tb,0,sizeof(tb)); FftGenerateTiling(soc,tb);     UploadGM(gm_fft_tiling,tb,1024);
      memset(tb,0,sizeof(tb)); LsGenerateTiling(soc,tb);      UploadGM(gm_ls_tiling,tb,1024);
      memset(tb,0,sizeof(tb)); LdpcGenerateTiling(soc,tb);    UploadGM(gm_ldpc_tiling,tb,8192);
      memset(tb,0,sizeof(tb)); DataExtGenerateTiling(soc,tb);  UploadGM(gm_dataext_tiling,tb,1024);
    }

    // Build permutation matrix for data subcarrier extraction
    { std::vector<uint16_t> perm(N_FFT * K_DATA_PAD, 0);
      auto sc = data_sc.data();
      int32_t n = std::min((int32_t)data_sc.size(), K_DATA_PAD);
      for (int32_t k = 0; k < n; k++) {
          int32_t row = sc[k];
          if (row >= 0 && row < N_FFT) perm[row * K_DATA_PAD + k] = 0x3C00; // half(1.0)
      }
      UploadGM(gm_perm_matrix, perm.data(), N_FFT * K_DATA_PAD * 2);
    }

    // Static tiling structs
    FftPostTiling fpt = {N_SYM, N_FFT};
    UploadGM(gm_fftpost_tiling, &fpt, sizeof(fpt));
    ExtractScTiling ept = {N_SYM, N_FFT, K_PILOT, 1, K_PILOT};
    UploadGM(gm_extract_pilot_tiling, &ept, sizeof(ept));
    ExtractScTiling edt = {N_SYM, N_FFT, K_DATA, 0, K_DATA_PAD};
    UploadGM(gm_extract_data_tiling, &edt, sizeof(edt));

    aclrtSynchronizeStream(g_stream);
    g_initialized = true;
    INFO_LOG("init_resources done.");
}

// ============================================================
// Timing storage — C++ stores, Python reads via get_timing()
// ============================================================
static struct {
    double h2d_cvt, h2d_dma;
    double sync_p1, sync_p2, sync_p3;
    double cfo, rrc, fft, eq, ext, qam, ldpc, decode_total;
} g_timing = {};

// ============================================================
// run_h2d — sc16 input, ARM NEON sc16→f16 conversion + DMA upload
// ============================================================
static void run_h2d(py::array_t<int16_t> sig_sc16) {
    auto t0 = clk::now();
    const int16_t *src = sig_sc16.data();
    size_t n_samples = sig_sc16.size() / 2;

    if (!g_r16) { g_r16 = new uint16_t[CAPTURE_LEN]; g_i16 = new uint16_t[CAPTURE_LEN]; }

    float32x4_t scale = vdupq_n_f32(1.0f / 32768.0f);
    size_t i = 0;
    for (; i + 4 <= n_samples; i += 4) {
        int16x4x2_t ri = vld2_s16(src + 2 * i);
        float32x4_t rf = vmulq_f32(vcvtq_f32_s32(vmovl_s16(ri.val[0])), scale);
        float32x4_t imf = vmulq_f32(vcvtq_f32_s32(vmovl_s16(ri.val[1])), scale);
        vst1_u16(g_r16 + i, vreinterpret_u16_f16(vcvt_f16_f32(rf)));
        vst1_u16(g_i16 + i, vreinterpret_u16_f16(vcvt_f16_f32(imf)));
    }
    for (; i < n_samples; i++) {
        __fp16 hr = (__fp16)((float)src[2*i] / 32768.0f);
        __fp16 hi = (__fp16)((float)src[2*i+1] / 32768.0f);
        memcpy(&g_r16[i], &hr, 2); memcpy(&g_i16[i], &hi, 2);
    }
    auto t1 = clk::now();

    UploadGM(gm_rx_r, g_r16, n_samples * 2);
    UploadGM(gm_rx_i, g_i16, n_samples * 2);
    aclrtSynchronizeStream(g_stream);

    g_timing.h2d_cvt = ms(t0, t1);
    g_timing.h2d_dma = ms(t1, clk::now());
}

// ============================================================
// run_sync — 3-launch parallel synchronization
// ============================================================
static py::tuple run_sync(int32_t search_start, int32_t search_len) {
    auto T = clk::now();
    float peak_host[64];

    auto fill_tiling = [&](FineSyncTiling &fst, int32_t ss, int32_t sl, int32_t nSym) {
        memset(&fst, 0, sizeof(fst));
        fst.searchStart = ss; fst.searchLen = sl; fst.refLen = ZC_UP;
        fst.totalN = CAPTURE_LEN; fst.nSym = nSym; fst.symUp = SYM_UP;
        fst.nFftUp = N_FFT * SPS; fst.cpUp = NCP * SPS;
        fst.nSymDecode = N_SYM; fst.zcUp = ZC_UP;
    };

    auto launch_sync = [&](FineSyncTiling &fst) {
        UploadGM(gm_sync_tiling, &fst, sizeof(fst));
        CHECK_ACL(aclrtMemset(gm_sync_peak, 64*4, 0, 64*4));
        ACLRT_LAUNCH_KERNEL(fine_sync)(BLOCK_DIM, g_stream,
            gm_rx_r, gm_rx_i, gm_sync_refR, gm_sync_refI,
            gm_sync_outR, gm_sync_outI, gm_sync_peak, gm_sync_tiling);
        aclrtSynchronizeStream(g_stream);
        DownloadGM(peak_host, gm_sync_peak, 64 * sizeof(float));
    };

    // Phase 1: parallel energy scan (searchStart=0)
    FineSyncTiling fst;
    fill_tiling(fst, 0, search_len, 0);
    launch_sync(fst);
    auto t1 = clk::now();

    int32_t coarse_pos = -1;
    for (int c = 0; c < 8; c++) {
        int32_t pos = (int32_t)peak_host[c*8]; float valid = peak_host[c*8+1];
        if (pos >= 0 && valid > 0.0f && (coarse_pos < 0 || pos < coarse_pos))
            coarse_pos = pos;
    }
    if (coarse_pos < 0) return py::make_tuple(-1, 0.0);

    // Phase 2: parallel cross-correlation (searchStart>0)
    int32_t margin = 512, ss = std::max(coarse_pos - margin, 1);
    fill_tiling(fst, ss, margin * 2 + ZC_UP, 0);
    launch_sync(fst);
    auto t2 = clk::now();

    int32_t fine_pos = -1; float best_pw = 0.0f;
    for (int c = 0; c < 8; c++) {
        int32_t pos = (int32_t)peak_host[c*8]; float pw = peak_host[c*8+1];
        if (pos >= 0 && pw > best_pw) { best_pw = pw; fine_pos = pos; }
    }
    if (fine_pos < 0) return py::make_tuple(-1, 0.0);

    // Phase 3: parallel CP-based CFO (searchStart=-1)
    fill_tiling(fst, -1, fine_pos, N_SYM);
    launch_sync(fst);
    auto t3 = clk::now();

    double Pr = 0, Pi = 0;
    for (int c = 0; c < 8; c++) { Pr += peak_host[c*8+2]; Pi += peak_host[c*8+3]; }
    double frac_cfo = -std::atan2(Pi, Pr) * 5e6 / (2.0 * M_PI * N_FFT * SPS);

    g_timing.sync_p1 = ms(T,t1); g_timing.sync_p2 = ms(t1,t2); g_timing.sync_p3 = ms(t2,t3);
    return py::make_tuple(fine_pos, frac_cfo);
}

// ============================================================
// run_decode — per-module timing for paper data
//   CFO → RRC → FFT → Channel Est → Data Ext → QAM → LDPC
// ============================================================
static py::array_t<int16_t> run_decode(
    int32_t fine_pos, double frac_cfo, int32_t n_ldpc_blocks)
{
    auto T = clk::now(), t0 = T;

    // --- CFO compensation ---
    uint32_t cfo_buf[CFO_TILING_WORDS]; memset(cfo_buf, 0, sizeof(cfo_buf));
    PrepareCfoTiling(cfo_buf, COMP_LEN, frac_cfo, 5e6, fine_pos);
    UploadGM(gm_cfo_tiling, cfo_buf, sizeof(cfo_buf));
    ACLRT_LAUNCH_KERNEL(cfo_compensate)(BLOCK_DIM, g_stream,
        (uint8_t*)gm_rx_r + fine_pos*2, (uint8_t*)gm_rx_i + fine_pos*2,
        gm_comp_r, gm_comp_i, gm_cfo_tiling);
    aclrtSynchronizeStream(g_stream);
    auto t_cfo = clk::now();

    // --- RRC matched filter + downsample ---
    CHECK_ACL(aclrtMemset(gm_rrc_r, N_SYM*SYM_TD*2, 0, N_SYM*SYM_TD*2));
    CHECK_ACL(aclrtMemset(gm_rrc_i, N_SYM*SYM_TD*2, 0, N_SYM*SYM_TD*2));
    ACLRT_LAUNCH_KERNEL(rrc_downsample)(BLOCK_DIM, g_stream,
        (uint8_t*)gm_comp_r+ZC_UP*2, (uint8_t*)gm_comp_i+ZC_UP*2,
        gm_rrc_coeff, gm_rrc_r, gm_rrc_i, gm_rrc_ws, gm_rrc_tiling);
    aclrtSynchronizeStream(g_stream);
    auto t_rrc = clk::now();

    // --- FFT (DFT cos + DFT sin + post-processing) ---
    ACLRT_LAUNCH_KERNEL(ofdm_fft)(BLOCK_DIM, g_stream,
        gm_rrc_r, gm_rrc_i, gm_fft_out0, gm_fft_out1,
        gm_fft_matB_cos, gm_dummy, gm_fft_ws, gm_fft_tiling);
    ACLRT_LAUNCH_KERNEL(ofdm_fft)(BLOCK_DIM, g_stream,
        gm_rrc_r, gm_rrc_i, gm_fft_out2, gm_fft_out3,
        gm_fft_matB_sin, gm_dummy, gm_fft_ws, gm_fft_tiling);
    ACLRT_LAUNCH_KERNEL(fft_postproc)(BLOCK_DIM, g_stream,
        gm_fft_out0, gm_fft_out1, gm_fft_out2, gm_fft_out3,
        gm_dc_r, gm_dc_i, gm_freq_r, gm_freq_i, gm_fftpost_tiling);
    aclrtSynchronizeStream(g_stream);
    auto t_fft = clk::now();

    // --- Channel estimation (pilot extraction + LS-ZF equalization) ---
    ACLRT_LAUNCH_KERNEL(extract_subcarriers)(BLOCK_DIM, g_stream,
        gm_freq_r, gm_freq_i, gm_pilot_sc, gm_pilot, gm_dummy, gm_dummy,
        gm_extract_pilot_tiling);
    ACLRT_LAUNCH_KERNEL(ls_equalizer)(BLOCK_DIM, g_stream,
        gm_pilot, gm_ls_matrix, gm_freq_r, gm_freq_i,
        gm_eq_r, gm_eq_i, gm_ls_ws, gm_ls_tiling);
    aclrtSynchronizeStream(g_stream);
    auto t_eq = clk::now();

    // --- Data subcarrier extraction ---
    ACLRT_LAUNCH_KERNEL(data_extract_mm)(BLOCK_DIM, g_stream,
        gm_eq_r, gm_eq_i, gm_data_r, gm_data_i,
        gm_perm_matrix, gm_dataext_ws, gm_dataext_tiling);
    aclrtSynchronizeStream(g_stream);
    auto t_ext = clk::now();

    // --- QAM demodulation ---
    ACLRT_LAUNCH_KERNEL(qam64_demod)(BLOCK_DIM, g_stream,
        gm_data_r, gm_data_i, gm_qam_out);
    aclrtSynchronizeStream(g_stream);
    auto t_qam = clk::now();

    // --- LDPC: D2H repack ---
    size_t qam_bytes = (size_t)N_SYM * QAM_OUT_COLS;
    std::vector<int8_t> qam_host(qam_bytes);
    DownloadGM(qam_host.data(), gm_qam_out, qam_bytes);
    std::vector<int8_t> ldpc_input(LDPC_ALLOC_CW * LDPC_N_CODEWORD, 0);
    { int32_t pos = 0;
      for (int32_t r = 0; r < N_SYM; r++) {
          memcpy(&ldpc_input[pos], &qam_host[r * QAM_OUT_COLS], K_DATA * 6);
          pos += K_DATA * 6;
      }
    }
    UploadGM(gm_ldpc_a, ldpc_input.data(), LDPC_ALLOC_CW * LDPC_N_CODEWORD);

    // --- LDPC: batch decode ---
    for (int32_t batch = 0; batch < LDPC_N_BATCHES; batch++) {
        size_t a_off = (size_t)batch * LDPC_BATCH * LDPC_N_CODEWORD;
        size_t info_off = (size_t)batch * LDPC_BATCH * 256 * sizeof(int16_t);
        CHECK_ACL(aclrtMemset(gm_ldpc_mask, 64*4, 0, 64*4));
        ACLRT_LAUNCH_KERNEL(ldpc_decode)(BLOCK_DIM, g_stream,
            (uint8_t*)gm_ldpc_a + a_off, gm_ldpc_b, gm_ldpc_c1, gm_ldpc_mask,
            gm_ldpc_c, gm_ldpc_ginv,
            (uint8_t*)gm_ldpc_info + info_off,
            gm_ldpc_ws, gm_ldpc_tiling);
    }
    aclrtSynchronizeStream(g_stream);

    // --- LDPC: D2H result ---
    std::vector<int16_t> all_info(LDPC_ALLOC_CW * 256);
    DownloadGM(all_info.data(), gm_ldpc_info, LDPC_ALLOC_CW * 256 * sizeof(int16_t));
    auto t_ldpc = clk::now();

    g_timing.cfo = ms(T,t_cfo); g_timing.rrc = ms(t_cfo,t_rrc);
    g_timing.fft = ms(t_rrc,t_fft); g_timing.eq = ms(t_fft,t_eq);
    g_timing.ext = ms(t_eq,t_ext); g_timing.qam = ms(t_ext,t_qam);
    g_timing.ldpc = ms(t_qam,t_ldpc); g_timing.decode_total = ms(T,t_ldpc);

    auto result = py::array_t<int16_t>(LDPC_TOTAL_CW * 256);
    memcpy(result.mutable_data(), all_info.data(), LDPC_TOTAL_CW * 256 * sizeof(int16_t));
    return result;
}

// ============================================================
// Utilities
// ============================================================
static void warmup() {
    run_decode(0, 0.0, 0); run_decode(0, 0.0, 0);
    INFO_LOG("Warmup done");
}

static py::array_t<uint16_t> dump_gm(const std::string &name, int32_t n_elems) {
    void *ptr = nullptr;
    if      (name == "comp_r")  ptr = gm_comp_r;
    else if (name == "comp_i")  ptr = gm_comp_i;
    else if (name == "rrc_r")   ptr = gm_rrc_r;
    else if (name == "rrc_i")   ptr = gm_rrc_i;
    else if (name == "freq_r")  ptr = gm_freq_r;
    else if (name == "freq_i")  ptr = gm_freq_i;
    else if (name == "eq_r")    ptr = gm_eq_r;
    else if (name == "eq_i")    ptr = gm_eq_i;
    else if (name == "pilot")   ptr = gm_pilot;
    else if (name == "data_r")  ptr = gm_data_r;
    else if (name == "data_i")  ptr = gm_data_i;
    else throw std::runtime_error("unknown buffer: " + name);
    auto result = py::array_t<uint16_t>(n_elems);
    DownloadGM(result.mutable_data(), ptr, n_elems * 2);
    return result;
}

static py::dict get_timing() {
    py::dict d;
    d["h2d_cvt"]  = g_timing.h2d_cvt;   d["h2d_dma"]  = g_timing.h2d_dma;
    d["sync_p1"]  = g_timing.sync_p1;    d["sync_p2"]  = g_timing.sync_p2;
    d["sync_p3"]  = g_timing.sync_p3;
    d["cfo"]      = g_timing.cfo;        d["rrc"]      = g_timing.rrc;
    d["fft"]      = g_timing.fft;        d["eq"]       = g_timing.eq;
    d["ext"]      = g_timing.ext;        d["qam"]      = g_timing.qam;
    d["ldpc"]     = g_timing.ldpc;       d["total"]    = g_timing.decode_total;
    return d;
}

static void cleanup() {
    if (g_stream) { aclrtDestroyStream(g_stream); g_stream = nullptr; }
    g_initialized = false;
}

PYBIND11_MODULE(ascend_baseband_rx_chain, m) {
    m.doc() = "Ascend-RAN: NPU-accelerated OFDM baseband processing";
    m.def("init_resources", &init_resources, "Allocate GM buffers and upload constants");
    m.def("run_h2d",        &run_h2d,        "sc16 → f16 conversion + H2D DMA");
    m.def("run_sync",       &run_sync,       "3-phase parallel frame synchronization");
    m.def("run_decode",     &run_decode,      "Full decode: CFO → RRC → FFT → EQ → QAM → LDPC");
    m.def("warmup",         &warmup,          "Warmup kernels (pre-allocate workspace)");
    m.def("get_timing",     &get_timing,      "Get last frame's per-module timing (ms)");
    m.def("dump_gm",        &dump_gm,         "Download GM buffer for debugging");
    m.def("cleanup",        &cleanup,          "Release resources");
}
