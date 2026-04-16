/**
 * @file pybind11.cpp — Ascend baseband processing platform (pybind interface)
 *
 * Exposes the complete OFDM TX chain to Python via pybind11:
 *   init_resources() — allocate GM buffers, upload constants
 *   run_tx()         — full TX: info bits → baseband signal (sample-interleaved)
 *
 * NPU kernel chain:
 *   LDPC encode → QAM64 modulation → OFDM IFFT (×2) → IFFT postproc
 *   → RRC upsample
 *
 * Optimizations:
 *   - LDPC async batching (13 launches, single sync)
 *   - QAM output directly in IFFT input stride (eliminates Pad step)
 *   - Dual IFFT workspace with pre-uploaded pilot bias (zero inter-sync)
 *   - ifft_postproc NPU kernel (replaces 4× D2H + host merge)
 *   - Bit pack on host ARM (cache-friendly, beats NPU scalar for stride-6)
 *   - NEON RMS + direct uint16 memcpy in RrcPrep
 *   - NEON vst4 host interleave (NPU scalar + PipeBarrier is 10× slower)
 *
 * Lesson learned: data rearrangement (bit pack, phase interleave) belongs on
 * ARM host with NEON, NOT on NPU. NPU scalar + PipeBarrier overhead per row
 * makes it significantly slower than vectorized host code for stride-N patterns.
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

#include "aclrtlaunch_ldpc_encode.h"
#include "aclrtlaunch_qam64_modulation.h"
#include "aclrtlaunch_ofdm_ifft.h"
#include "aclrtlaunch_ifft_postproc.h"
#include "aclrtlaunch_up_sample_rrc.h"

extern "C" void LdpcEncGenerateTiling(const char *socVersion, uint8_t *tilingBuf);
extern "C" void IfftGenerateTiling(const char *socVersion, uint8_t *tilingBuf);
extern "C" void RrcUpGenerateTiling(const char *socVersion, uint8_t *tilingBuf);

namespace py = pybind11;

// ============================================================
// System constants
// ============================================================
static constexpr int32_t BLOCK_DIM   = 8;
static constexpr int32_t N_FFT       = 256;
static constexpr int32_t NCP         = 16;
static constexpr int32_t SPS         = 4;
static constexpr int32_t N_SYM       = 1192;
static constexpr int32_t SYM_TD      = N_FFT + NCP;           // 272
static constexpr int32_t K_DATA      = 220;
static constexpr int32_t K_IN        = 224;                    // 32B-aligned IFFT input stride
static constexpr int32_t ZC_N        = 256;

// LDPC
static constexpr int32_t LDPC_K          = 256;
static constexpr int32_t LDPC_N          = 512;
static constexpr int32_t LDPC_BATCH      = 256;
static constexpr int32_t LDPC_TOTAL_BITS = N_SYM * K_DATA * 6;           // 1573440
static constexpr int32_t LDPC_TOTAL_CW   = LDPC_TOTAL_BITS / LDPC_N;     // 3073
static constexpr int32_t LDPC_N_BATCHES  = (LDPC_TOTAL_CW + LDPC_BATCH - 1) / LDPC_BATCH;
static constexpr int32_t LDPC_ALLOC_CW   = LDPC_N_BATCHES * LDPC_BATCH;  // 3328

// QAM
static constexpr int32_t QAM_TOTAL        = N_SYM * K_DATA;              // 262240
static constexpr int32_t QAM_FROM_LDPC    = LDPC_TOTAL_CW * LDPC_N / 6;  // 262229

// RRC
static constexpr int32_t RRC_STRIDE         = 256;
static constexpr int32_t RRC_OVERLAP        = SYM_TD - RRC_STRIDE;       // 16
static constexpr int32_t RRC_MAX_ROWS       = 1300;
static constexpr int32_t RRC_TILING_ALLOC   = 768;
static constexpr int32_t RRC_TOTAL_ROWS_OFF = 512;

// IFFT workspace
static constexpr int32_t IFFT_BIAS_OFFSET = 512 * 2;                     // bytes
static constexpr int32_t IFFT_WS_SIZE     = 8 * 1024 * 1024;

// ============================================================
// Global state
// ============================================================
static aclrtStream g_stream = nullptr;
static bool g_initialized = false;
static bool g_bias_uploaded = false;
static uint8_t *g_rrc_tiling_host = nullptr;

// GM buffer pointers (allocated in init_resources)
static void *gm_ldpc_a, *gm_ldpc_G, *gm_ldpc_c, *gm_ldpc_ws, *gm_ldpc_tiling;
static void *gm_qam_bits, *gm_qam_r, *gm_qam_i;
static void *gm_ifft_in_r, *gm_ifft_in_i;
static void *gm_ifft_out0, *gm_ifft_out1, *gm_ifft_out2, *gm_ifft_out3;
static void *gm_ifft_matB_cos, *gm_ifft_matB_sin;
static void *gm_ifft_ws_cos, *gm_ifft_ws_sin, *gm_ifft_mmws;
static void *gm_td_r, *gm_td_i, *gm_postproc_tiling;
static void *gm_rrc_in_r, *gm_rrc_in_i, *gm_rrc_coeff;
static void *gm_rrc_out_r, *gm_rrc_out_i, *gm_rrc_ws, *gm_rrc_tiling;
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

// ============================================================
// init_resources — allocate all GM buffers and upload constants
// ============================================================
static void init_resources(
    py::array_t<int8_t>   ldpc_G,
    py::array_t<uint16_t> ifft_cos, py::array_t<uint16_t> ifft_sin,
    py::array_t<uint16_t> rrc_coeff)
{
    if (g_initialized) return;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateStream(&g_stream));
    const char *soc = "Ascend310B1";

    // LDPC encode (large buffers for async batching)
    gm_ldpc_a      = AllocGM(LDPC_ALLOC_CW * LDPC_K);
    gm_ldpc_G      = AllocGM(LDPC_K * LDPC_N);
    gm_ldpc_c      = AllocGM(LDPC_ALLOC_CW * LDPC_N * 2);
    gm_ldpc_ws     = AllocGM(4 * 1024 * 1024);
    gm_ldpc_tiling = AllocGM(4096);
    UploadGM(gm_ldpc_G, ldpc_G.data(), LDPC_K * LDPC_N);
    { uint8_t tb[4096] = {}; LdpcEncGenerateTiling(soc, tb);
      UploadGM(gm_ldpc_tiling, tb, 4096); }

    // QAM bits input + output aliased to IFFT input (stride = K_IN)
    gm_qam_bits  = AllocGM(QAM_TOTAL);
    gm_ifft_in_r = AllocGM(N_SYM * K_IN * 2);
    gm_ifft_in_i = AllocGM(N_SYM * K_IN * 2);
    gm_qam_r     = gm_ifft_in_r;   // QAM writes directly to IFFT input GM
    gm_qam_i     = gm_ifft_in_i;

    // IFFT outputs (4 buffers for cos/sin × R/I)
    gm_ifft_out0     = AllocGM(N_SYM * SYM_TD * 2);
    gm_ifft_out1     = AllocGM(N_SYM * SYM_TD * 2);
    gm_ifft_out2     = AllocGM(N_SYM * SYM_TD * 2);
    gm_ifft_out3     = AllocGM(N_SYM * SYM_TD * 2);
    gm_ifft_matB_cos = AllocGM(ifft_cos.size() * 2);
    gm_ifft_matB_sin = AllocGM(ifft_sin.size() * 2);
    UploadGM(gm_ifft_matB_cos, ifft_cos.data(), ifft_cos.size() * 2);
    UploadGM(gm_ifft_matB_sin, ifft_sin.data(), ifft_sin.size() * 2);

    // Dual IFFT workspace (pre-uploaded tiling; bias uploaded once per session)
    gm_ifft_ws_cos = AllocGM(IFFT_WS_SIZE);
    gm_ifft_ws_sin = AllocGM(IFFT_WS_SIZE);
    { uint8_t tb[1024] = {}; IfftGenerateTiling(soc, tb);
      UploadGM(gm_ifft_ws_cos, tb, 1024);
      UploadGM(gm_ifft_ws_sin, tb, 1024); }
    gm_ifft_mmws = AllocGM(4 * 1024 * 1024);

    // IFFT postproc output (time-domain baseband)
    gm_td_r            = AllocGM(N_SYM * SYM_TD * 2);
    gm_td_i            = AllocGM(N_SYM * SYM_TD * 2);
    gm_postproc_tiling = AllocGM(64);
    { int32_t pp[2] = {N_SYM, SYM_TD};
      UploadGM(gm_postproc_tiling, pp, sizeof(pp)); }

    // RRC polyphase upsampling
    size_t rrc_sz = (size_t)RRC_MAX_ROWS * SYM_TD * 2;
    gm_rrc_in_r   = AllocGM(rrc_sz);
    gm_rrc_in_i   = AllocGM(rrc_sz);
    gm_rrc_coeff  = AllocGM(rrc_coeff.size() * 2);
    gm_rrc_out_r  = AllocGM(SPS * rrc_sz);
    gm_rrc_out_i  = AllocGM(SPS * rrc_sz);
    gm_rrc_ws     = AllocGM(4 * 1024 * 1024);
    gm_rrc_tiling = AllocGM(RRC_TILING_ALLOC);
    UploadGM(gm_rrc_coeff, rrc_coeff.data(), rrc_coeff.size() * 2);
    g_rrc_tiling_host = new uint8_t[RRC_TILING_ALLOC]();
    RrcUpGenerateTiling(soc, g_rrc_tiling_host);
    UploadGM(gm_rrc_tiling, g_rrc_tiling_host, RRC_TILING_ALLOC);

    gm_dummy = AllocGM(64);
    g_bias_uploaded = false;
    aclrtSynchronizeStream(g_stream);
    g_initialized = true;
    INFO_LOG("TX init_resources done.");
}

// ============================================================
// Timing storage — C++ stores, Python reads via get_timing()
// ============================================================
static struct {
    double ldpc, bit_pack, qam, ifft, postproc, rrc_prep, rrc, interleave, total;
} g_timing = {};

// ============================================================
// run_tx — full TX pipeline
// ============================================================
static py::tuple run_tx(
    py::array_t<int8_t>   info_bits,
    py::array_t<uint16_t> pilot_bias_r,
    py::array_t<uint16_t> pilot_bias_i,
    py::array_t<uint16_t> zc_r,
    py::array_t<uint16_t> zc_i)
{
    auto T = clk::now(), t0 = T;
    int32_t total_cw = info_bits.shape(0);
    int32_t n_batches = (total_cw + LDPC_BATCH - 1) / LDPC_BATCH;

    // --- LDPC encode (async batching, single final sync) ---
    std::vector<int8_t> ldpc_in(LDPC_ALLOC_CW * LDPC_K, 0);
    memcpy(ldpc_in.data(), info_bits.data(), total_cw * LDPC_K);
    UploadGM(gm_ldpc_a, ldpc_in.data(), LDPC_ALLOC_CW * LDPC_K);
    for (int32_t b = 0; b < n_batches; b++) {
        ACLRT_LAUNCH_KERNEL(ldpc_encode)(BLOCK_DIM, g_stream,
            (uint8_t*)gm_ldpc_a + (size_t)b * LDPC_BATCH * LDPC_K,
            gm_ldpc_G,
            (uint8_t*)gm_ldpc_c + (size_t)b * LDPC_BATCH * LDPC_N * 2,
            gm_ldpc_ws, gm_ldpc_tiling);
    }
    aclrtSynchronizeStream(g_stream);
    auto t_ldpc = clk::now();

    // --- Bit pack (host): int16 codewords → uint8 6-bit QAM bytes ---
    {
        std::vector<int16_t> cw(LDPC_ALLOC_CW * LDPC_N);
        DownloadGM(cw.data(), gm_ldpc_c, (size_t)total_cw * LDPC_N * sizeof(int16_t));
        std::vector<uint8_t> qb(QAM_TOTAL, 0);
        const int16_t *c = cw.data();
        uint8_t *out = qb.data();
        for (int32_t i = 0; i < QAM_FROM_LDPC; i++, c += 6) {
            *out++ = ((c[0]&1)<<5) | ((c[1]&1)<<4) | ((c[2]&1)<<3)
                   | ((c[3]&1)<<2) | ((c[4]&1)<<1) |  (c[5]&1);
        }
        UploadGM(gm_qam_bits, qb.data(), QAM_TOTAL);
    }
    auto t_pack = clk::now();

    // --- QAM64 modulation (writes directly to IFFT input GM, stride = K_IN) ---
    ACLRT_LAUNCH_KERNEL(qam64_modulation)(BLOCK_DIM, g_stream,
        gm_qam_bits, gm_qam_r, gm_qam_i);
    aclrtSynchronizeStream(g_stream);
    auto t_qam = clk::now();

    // --- OFDM IFFT ×2 (dual workspace, zero inter-sync) ---
    // Pilot bias uploaded once (constant across frames)
    if (!g_bias_uploaded) {
        UploadGM((uint8_t*)gm_ifft_ws_cos + IFFT_BIAS_OFFSET,
                 pilot_bias_r.data(), pilot_bias_r.size() * 2);
        UploadGM((uint8_t*)gm_ifft_ws_sin + IFFT_BIAS_OFFSET,
                 pilot_bias_i.data(), pilot_bias_i.size() * 2);
        g_bias_uploaded = true;
    }
    CHECK_ACL(aclrtMemset(gm_ifft_mmws, 4*1024*1024, 0, 4*1024*1024));
    ACLRT_LAUNCH_KERNEL(ofdm_ifft)(BLOCK_DIM, g_stream,
        gm_ifft_in_r, gm_ifft_in_i, gm_ifft_out0, gm_ifft_out1,
        gm_ifft_matB_cos, gm_dummy, gm_ifft_ws_cos, gm_ifft_mmws);
    ACLRT_LAUNCH_KERNEL(ofdm_ifft)(BLOCK_DIM, g_stream,
        gm_ifft_in_r, gm_ifft_in_i, gm_ifft_out2, gm_ifft_out3,
        gm_ifft_matB_sin, gm_dummy, gm_ifft_ws_sin, gm_ifft_mmws);
    aclrtSynchronizeStream(g_stream);
    auto t_ifft = clk::now();

    // --- IFFT post-processing: td_r = o0-o3, td_i = o1+o2 ---
    ACLRT_LAUNCH_KERNEL(ifft_postproc)(BLOCK_DIM, g_stream,
        gm_ifft_out0, gm_ifft_out1, gm_ifft_out2, gm_ifft_out3,
        gm_td_r, gm_td_i, gm_postproc_tiling);
    aclrtSynchronizeStream(g_stream);
    auto t_post = clk::now();

    // --- RRC prep: D2H + ZC prepend + overlap row construction ---
    size_t td_elems = (size_t)N_SYM * SYM_TD;
    std::vector<uint16_t> tdR(td_elems), tdI(td_elems);
    DownloadGM(tdR.data(), gm_td_r, td_elems * 2);
    DownloadGM(tdI.data(), gm_td_i, td_elems * 2);

    // NEON-vectorized sum of squares for RMS computation
    double sum_sq;
    {
        float32x4_t acc = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= td_elems; i += 4) {
            float32x4_t fr = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&tdR[i])));
            float32x4_t fi = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(&tdI[i])));
            acc = vmlaq_f32(acc, fr, fr);
            acc = vmlaq_f32(acc, fi, fi);
        }
        sum_sq = (double)vaddvq_f32(acc);
        for (; i < td_elems; i++) {
            __fp16 vr, vi; memcpy(&vr, &tdR[i], 2); memcpy(&vi, &tdI[i], 2);
            sum_sq += (float)vr * (float)vr + (float)vi * (float)vi;
        }
    }
    double data_rms = std::sqrt(sum_sq / td_elems);
    if (data_rms < 1e-10) data_rms = 1e-10;

    // ZC scaling: boost sync sequence to 2× data RMS
    double zc_sq = 0.0;
    auto *zrp = reinterpret_cast<const __fp16*>(zc_r.data());
    auto *zip = reinterpret_cast<const __fp16*>(zc_i.data());
    for (int32_t i = 0; i < ZC_N; i++) {
        float r = (float)zrp[i], im = (float)zip[i];
        zc_sq += r * r + im * im;
    }
    float zc_scale = (float)(2.0 * data_rms / std::sqrt(zc_sq / ZC_N));

    // Build overlap-add row layout: [pad | ZC | td_data]
    int32_t TD_START = RRC_OVERLAP + ZC_N;
    int32_t L_padded = TD_START + N_SYM * SYM_TD;
    int32_t n_rows = (L_padded - SYM_TD) / RRC_STRIDE + 1;
    if ((n_rows - 1) * RRC_STRIDE + SYM_TD < L_padded) n_rows++;

    std::vector<uint16_t> rrcR(n_rows * SYM_TD, 0), rrcI(n_rows * SYM_TD, 0);

    // Row 0: ZC sequence (scaled, float conversion required)
    for (int32_t i = 0; i < ZC_N; i++) {
        __fp16 hr = (__fp16)((float)zrp[i] * zc_scale);
        __fp16 hi = (__fp16)((float)zip[i] * zc_scale);
        memcpy(&rrcR[RRC_OVERLAP + i], &hr, 2);
        memcpy(&rrcI[RRC_OVERLAP + i], &hi, 2);
    }
    // Row 1: last 16 ZC samples + first 256 td samples
    for (int32_t i = 0; i < RRC_OVERLAP; i++) {
        int32_t zi = RRC_STRIDE - RRC_OVERLAP + i;
        if (zi < ZC_N) {
            __fp16 hr = (__fp16)((float)zrp[zi] * zc_scale);
            __fp16 hi = (__fp16)((float)zip[zi] * zc_scale);
            memcpy(&rrcR[SYM_TD + i], &hr, 2);
            memcpy(&rrcI[SYM_TD + i], &hi, 2);
        }
    }
    memcpy(&rrcR[SYM_TD + RRC_OVERLAP], &tdR[0], RRC_STRIDE * 2);
    memcpy(&rrcI[SYM_TD + RRC_OVERLAP], &tdI[0], RRC_STRIDE * 2);

    // Rows 2+: pure td data (direct uint16 memcpy, zero float conversion)
    for (int32_t r = 2; r < n_rows; r++) {
        int32_t td_off = r * RRC_STRIDE - TD_START;
        int32_t len = SYM_TD;
        if (td_off + len > (int32_t)td_elems) len = (int32_t)td_elems - td_off;
        if (td_off >= 0 && len > 0) {
            memcpy(&rrcR[r * SYM_TD], &tdR[td_off], len * 2);
            memcpy(&rrcI[r * SYM_TD], &tdI[td_off], len * 2);
        }
    }

    *reinterpret_cast<int32_t*>(g_rrc_tiling_host + RRC_TOTAL_ROWS_OFF) = n_rows;
    UploadGM(gm_rrc_tiling, g_rrc_tiling_host, RRC_TILING_ALLOC);
    UploadGM(gm_rrc_in_r, rrcR.data(), n_rows * SYM_TD * 2);
    UploadGM(gm_rrc_in_i, rrcI.data(), n_rows * SYM_TD * 2);
    aclrtSynchronizeStream(g_stream);
    auto t_prep = clk::now();

    // --- RRC upsample (polyphase Toeplitz Matmul, 4 phases × R/I) ---
    ACLRT_LAUNCH_KERNEL(up_sample_rrc)(BLOCK_DIM, g_stream,
        gm_rrc_in_r, gm_rrc_in_i, gm_rrc_coeff,
        gm_rrc_out_r, gm_rrc_out_i, gm_rrc_ws, gm_rrc_tiling);
    aclrtSynchronizeStream(g_stream);
    auto t_rrc = clk::now();

    // --- D2H + NEON 4-phase interleave (host) ---
    size_t rrc_elems = (size_t)SPS * n_rows * SYM_TD;
    std::vector<uint16_t> outR(rrc_elems), outI(rrc_elems);
    DownloadGM(outR.data(), gm_rrc_out_r, rrc_elems * 2);
    DownloadGM(outI.data(), gm_rrc_out_i, rrc_elems * 2);

    int32_t bb_total = n_rows * RRC_STRIDE * SPS;
    auto result_r = py::array_t<uint16_t>(bb_total);
    auto result_i = py::array_t<uint16_t>(bb_total);
    uint16_t *rr = result_r.mutable_data();
    uint16_t *ri = result_i.mutable_data();

    size_t pstride = (size_t)n_rows * SYM_TD;
    for (int32_t row = 0; row < n_rows; row++) {
        uint16_t *s0r = &outR[0*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *s1r = &outR[1*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *s2r = &outR[2*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *s3r = &outR[3*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *dr  = &rr[row * RRC_STRIDE * SPS];
        uint16_t *s0i = &outI[0*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *s1i = &outI[1*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *s2i = &outI[2*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *s3i = &outI[3*pstride + row*SYM_TD + RRC_OVERLAP];
        uint16_t *di  = &ri[row * RRC_STRIDE * SPS];

        int32_t s = 0;
        for (; s + 8 <= RRC_STRIDE; s += 8) {
            uint16x8x4_t vr = {vld1q_u16(s0r+s), vld1q_u16(s1r+s),
                                vld1q_u16(s2r+s), vld1q_u16(s3r+s)};
            vst4q_u16(dr + s*4, vr);
            uint16x8x4_t vi = {vld1q_u16(s0i+s), vld1q_u16(s1i+s),
                                vld1q_u16(s2i+s), vld1q_u16(s3i+s)};
            vst4q_u16(di + s*4, vi);
        }
        for (; s < RRC_STRIDE; s++) {
            dr[s*4]   = s0r[s]; dr[s*4+1] = s1r[s];
            dr[s*4+2] = s2r[s]; dr[s*4+3] = s3r[s];
            di[s*4]   = s0i[s]; di[s*4+1] = s1i[s];
            di[s*4+2] = s2i[s]; di[s*4+3] = s3i[s];
        }
    }
    auto t_intlv = clk::now();

    g_timing.ldpc       = ms(T,      t_ldpc);
    g_timing.bit_pack   = ms(t_ldpc, t_pack);
    g_timing.qam        = ms(t_pack, t_qam);
    g_timing.ifft       = ms(t_qam,  t_ifft);
    g_timing.postproc   = ms(t_ifft, t_post);
    g_timing.rrc_prep   = ms(t_post, t_prep);
    g_timing.rrc        = ms(t_prep, t_rrc);
    g_timing.interleave = ms(t_rrc,  t_intlv);
    g_timing.total      = ms(T,      t_intlv);

    return py::make_tuple(result_r, result_i, n_rows);
}

// ============================================================
// Utilities
// ============================================================
static void warmup(
    py::array_t<int8_t>   info_bits,
    py::array_t<uint16_t> pilot_r, py::array_t<uint16_t> pilot_i,
    py::array_t<uint16_t> zc_r,    py::array_t<uint16_t> zc_i)
{
    run_tx(info_bits, pilot_r, pilot_i, zc_r, zc_i);
    run_tx(info_bits, pilot_r, pilot_i, zc_r, zc_i);
    INFO_LOG("Warmup done");
}

static py::array_t<uint16_t> dump_gm(const std::string &name, int32_t n_elems) {
    void *ptr = nullptr;
    if      (name == "ldpc_c")   ptr = gm_ldpc_c;
    else if (name == "qam_r")    ptr = gm_qam_r;
    else if (name == "qam_i")    ptr = gm_qam_i;
    else if (name == "ifft_in_r") ptr = gm_ifft_in_r;
    else if (name == "ifft_in_i") ptr = gm_ifft_in_i;
    else if (name == "td_r")     ptr = gm_td_r;
    else if (name == "td_i")     ptr = gm_td_i;
    else if (name == "rrc_out_r") ptr = gm_rrc_out_r;
    else if (name == "rrc_out_i") ptr = gm_rrc_out_i;
    else throw std::runtime_error("unknown buffer: " + name);
    auto result = py::array_t<uint16_t>(n_elems);
    DownloadGM(result.mutable_data(), ptr, n_elems * 2);
    return result;
}

static py::dict get_timing() {
    py::dict d;
    d["ldpc"]       = g_timing.ldpc;
    d["bit_pack"]   = g_timing.bit_pack;
    d["qam"]        = g_timing.qam;
    d["ifft"]       = g_timing.ifft;
    d["postproc"]   = g_timing.postproc;
    d["rrc_prep"]   = g_timing.rrc_prep;
    d["rrc"]        = g_timing.rrc;
    d["interleave"] = g_timing.interleave;
    d["total"]      = g_timing.total;
    return d;
}

static void cleanup() {
    if (g_rrc_tiling_host) { delete[] g_rrc_tiling_host; g_rrc_tiling_host = nullptr; }
    if (g_stream) { aclrtDestroyStream(g_stream); g_stream = nullptr; }
    g_initialized = false;
    g_bias_uploaded = false;
}

PYBIND11_MODULE(ascend_baseband_tx_chain, m) {
    m.doc() = "Ascend-RAN: NPU-accelerated OFDM baseband processing (TX)";
    m.def("init_resources", &init_resources, "Allocate GM buffers and upload constants");
    m.def("run_tx",         &run_tx,         "Full TX: info bits → baseband signal");
    m.def("warmup",         &warmup,         "Warmup kernels (pre-allocate workspace)");
    m.def("get_timing",     &get_timing,     "Get last frame's per-module timing (ms)");
    m.def("dump_gm",        &dump_gm,        "Download GM buffer for debugging");
    m.def("cleanup",        &cleanup,        "Release resources");
}