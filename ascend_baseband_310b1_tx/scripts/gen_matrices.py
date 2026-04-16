#!/usr/bin/env python3
"""
gen_matrices.py — Generate all precomputed matrices for OFDM ISAC system

Generates constant data files required by the NPU RX/TX chain:
  [1] IFFT DFT matrix + pilot bias          (TX)
  [2] FFT DFT matrix + subcarrier indices   (RX)
  [3] RRC polyphase Toeplitz matrices        (TX upsampling)
  [4] RRC downsample + CP removal matrix     (RX)
  [5] ZC synchronization reference sequence  (RX sync)
  [6] LS channel estimation matrix           (RX equalization)
  [7] FIR anti-aliasing Toeplitz matrix      (RX, optional)
  [8] LDPC parity check / generator matrices (RX/TX, PEG construction)

Usage:
  python3 gen_matrices.py                     # generate all
  python3 gen_matrices.py --skip-ldpc         # skip LDPC (slow)
  python3 gen_matrices.py --output_dir ./input

Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
SPDX-License-Identifier: MIT
"""
import numpy as np
import os
import argparse


# ============================================================
# System parameters
# ============================================================
NFFT  = 256
NCP   = 16
SPS   = 4
SPAN  = 8
BETA  = 0.35
N_OUT  = NFFT + NCP    # 272 (symbol length with CP)
SYM_IN = N_OUT * SPS   # 1088 (upsampled symbol length)

# Subcarrier allocation (MATLAB 1-indexed -> 0-indexed)
PROTECT_MATLAB = [x + 129 for x in
    list(range(-128, -121 + 1)) + list(range(-2, 2 + 1)) + list(range(121, 127 + 1))]
PILOT_MATLAB = [x + 129 for x in
    [-120, -104, -88, -72, -56, -40, -24, -8, 8, 24, 40, 56, 72, 88, 104, 120]]
DATA_MATLAB = sorted(set(range(1, NFFT + 1)) - set(PROTECT_MATLAB) - set(PILOT_MATLAB))

DATA_IDX  = [s - 1 for s in DATA_MATLAB]     # 0-indexed
PILOT_IDX = [s - 1 for s in PILOT_MATLAB]
GUARD_IDX = [s - 1 for s in PROTECT_MATLAB]

N_DATA  = len(DATA_IDX)    # 220
N_PILOT = len(PILOT_IDX)   # 16
N_GUARD = len(GUARD_IDX)   # 20

# LDPC parameters
LDPC_M = 256              # information bits
LDPC_N = 256              # parity bits
LDPC_K = 512              # codeword length
LDPC_COL_WEIGHT = 3
LDPC_SEED = 42

# ZC sequence
ZC_NZC = 251              # largest prime <= 256
ZC_U   = 1                # root index
ZC_REF_LEN = 1024         # upsampled length = 256 * SPS


# ============================================================
# RRC filter
# ============================================================
def rrc_filter(beta, span, sps):
    """Root raised cosine filter impulse response (unit energy)."""
    N = 2 * span * sps
    t = np.arange(-N // 2, N // 2 + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-10 and beta > 0:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


# ============================================================
# [1] IFFT matrix + pilot bias (TX)
# ============================================================
def gen_ifft_matrix(output_dir):
    """IFFT DFT matrix with built-in CP insertion + pilot bias vector.

    G[n, k] = exp(j*2*pi*t_n*f_k / NFFT), where t_n = (n-NCP) mod NFFT
    bias[n] = (1/NFFT) * sum(pilot_val * exp(j*2*pi*t_n*pilot_pos / NFFT))
    """
    print(f"\n[1] IFFT matrix [{N_OUT}, {N_OUT}] + pilot bias [{N_OUT}]")

    G = np.zeros((N_OUT, N_OUT), dtype=np.complex128)
    for n in range(N_OUT):
        t = (n - NCP) % NFFT
        for k_idx, freq_bin in enumerate(DATA_IDX):
            G[n, k_idx] = np.exp(1j * 2 * np.pi * t * freq_bin / NFFT)

    # Pilot value (matches TX SubcarrierMapper: max QAM64 level * (1+1j))
    SCALE_l = 2.0 / np.sqrt(42)
    OFFSET_l = -7.0 / np.sqrt(42)
    LEVELS_l = np.arange(8) * SCALE_l + OFFSET_l
    pilot_val = np.max(np.abs(LEVELS_l)) * (1 + 1j)

    bias = np.zeros(N_OUT, dtype=np.complex128)
    for n in range(N_OUT):
        t = (n - NCP) % NFFT
        for p_idx, freq_bin in enumerate(PILOT_IDX):
            bias[n] += pilot_val * np.exp(1j * 2 * np.pi * t * freq_bin / NFFT)
    bias /= NFFT

    G.real.astype(np.float16).tofile(os.path.join(output_dir, 'matG_ifft_real.bin'))
    G.imag.astype(np.float16).tofile(os.path.join(output_dir, 'matG_ifft_imag.bin'))
    bias.real.astype(np.float16).tofile(os.path.join(output_dir, 'pilot_bias_real.bin'))
    bias.imag.astype(np.float16).tofile(os.path.join(output_dir, 'pilot_bias_imag.bin'))

    # Verify against standard IFFT
    np.random.seed(42)
    qam = (np.random.randn(4, N_DATA) + 1j * np.random.randn(4, N_DATA)).astype(np.complex64)
    freq = np.zeros((4, NFFT), dtype=np.complex64)
    for k_idx, mi in enumerate(DATA_MATLAB):
        freq[:, mi - 1] = qam[:, k_idx]
    for p_idx, mi in enumerate(PILOT_MATLAB):
        freq[:, mi - 1] = pilot_val
    td = np.fft.ifft(freq, axis=1)
    ofdm_std = np.concatenate([td[:, -NCP:], td], axis=1)

    qam_pad = np.zeros((4, N_OUT), dtype=np.complex128)
    qam_pad[:, :N_DATA] = qam
    ofdm_mat = (qam_pad @ G.T) / NFFT + bias
    err = np.abs(ofdm_std - ofdm_mat).max()
    print(f"    Verify: max_err={err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")


# ============================================================
# [2] FFT matrix + subcarrier indices (RX)
# ============================================================
def gen_fft_matrix(output_dir):
    """FFT DFT matrix with built-in CP removal.

    G[k, t] = exp(-j*2*pi*(t-NCP)*k / NFFT) for t >= NCP, else 0
    """
    print(f"\n[2] FFT matrix [{NFFT}, {N_OUT}]")

    G = np.zeros((NFFT, N_OUT), dtype=np.complex64)
    for k in range(NFFT):
        for t in range(NCP, N_OUT):
            G[k, t] = np.exp(-1j * 2 * np.pi * (t - NCP) * k / NFFT)

    G.real.astype(np.float16).tofile(os.path.join(output_dir, 'matG_fft_real.bin'))
    G.imag.astype(np.float16).tofile(os.path.join(output_dir, 'matG_fft_imag.bin'))
    np.array(DATA_IDX,  dtype=np.int32).tofile(os.path.join(output_dir, 'data_subcarrier_idx.bin'))
    np.array(PILOT_IDX, dtype=np.int32).tofile(os.path.join(output_dir, 'pilot_subcarrier_idx.bin'))
    np.array(GUARD_IDX, dtype=np.int32).tofile(os.path.join(output_dir, 'guard_subcarrier_idx.bin'))

    # Verify
    np.random.seed(123)
    td = (np.random.randn(4, N_OUT) + 1j * np.random.randn(4, N_OUT)).astype(np.complex64)
    freq_std = np.fft.fft(td[:, NCP:], axis=1)
    freq_mat = td @ G.T
    err = np.abs(freq_std - freq_mat).max()
    print(f"    Verify: max_err={err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")


# ============================================================
# [3] RRC polyphase Toeplitz matrices (TX upsampling)
# ============================================================
def gen_rrc_toeplitz(output_dir):
    """Generate 4 polyphase Toeplitz matrices for RRC upsampling.

    H_p[i, j] = h_p[i-j], p = 0..3, each [272, 272]
    """
    print(f"\n[3] RRC Toeplitz 4 x [{N_OUT}, {N_OUT}]")

    h = rrc_filter(BETA, SPAN, SPS)
    print(f"    RRC filter: {len(h)} taps")

    for p in range(SPS):
        hp = h[p::SPS]
        L = len(hp)
        H = np.zeros((N_OUT, N_OUT), dtype=np.float64)
        for i in range(N_OUT):
            for j in range(N_OUT):
                if 0 <= i - j < L:
                    H[i, j] = hp[i - j]
        H.astype(np.float16).tofile(os.path.join(output_dir, f'rrc_toeplitz_h{p}.bin'))
        print(f"    H{p}: max={np.abs(H).max():.6f}, energy={np.sum(hp**2):.6f}")

    h.astype(np.float64).tofile(os.path.join(output_dir, 'rrc_filter.bin'))

    # Verify Toeplitz vs direct convolution
    np.random.seed(42)
    x = np.random.randn(10, N_OUT)
    x_conv = np.zeros((10, SYM_IN), dtype=np.float64)
    for s in range(10):
        row_up = np.zeros(SYM_IN)
        row_up[::SPS] = x[s]
        x_conv[s] = np.convolve(row_up, h, mode='full')[:SYM_IN]

    x_toepl = np.zeros((10, SYM_IN), dtype=np.float64)
    for p in range(SPS):
        hp = h[p::SPS]
        L = len(hp)
        Hp = np.zeros((N_OUT, N_OUT), dtype=np.float64)
        for i in range(N_OUT):
            for j in range(N_OUT):
                if 0 <= i - j < L:
                    Hp[i, j] = hp[i - j]
        x_toepl[:, p::SPS] = x @ Hp.T

    err = np.abs(x_conv - x_toepl).max()
    print(f"    Verify: Toeplitz vs convolve max_err={err:.2e}  {'PASS' if err < 1e-10 else 'FAIL'}")


# ============================================================
# [4] RRC downsample + CP removal fused matrix (RX)
# ============================================================
def gen_rrc_down_matrix(output_dir):
    """Fused RRC matched filter + downsample + CP removal.

    T[j, n] = h[(NCP+n)*SPS + delay - j], shape [1088, 256]
    Also generates T272 [1088, 272] with CP zero-padding for stride-272 layout.
    """
    print(f"\n[4] RRC down matrix T [{SYM_IN}, {NFFT}]")

    h = rrc_filter(BETA, SPAN, SPS)
    delay = len(h) // 2

    T = np.zeros((SYM_IN, NFFT), dtype=np.float64)
    for n in range(NFFT):
        for j in range(SYM_IN):
            idx = (NCP + n) * SPS + delay - j
            if 0 <= idx < len(h):
                T[j, n] = h[idx]

    T.astype(np.float16).tofile(os.path.join(output_dir, 'rrc_down_T.bin'))
    print(f"    max={np.abs(T).max():.6f}, filter_delay={delay}")

    # T272: zero-padded to stride-272 layout (CP columns are zeros)
    T272 = np.zeros((SYM_IN, N_OUT), dtype=np.float64)
    T272[:, NCP:] = T
    T272.astype(np.float16).tofile(os.path.join(output_dir, 'rrc_down_T272.bin'))
    print(f"    + rrc_down_T272.bin [{SYM_IN}, {N_OUT}] ({T272.astype(np.float16).nbytes/1024:.1f} KB)")

    # Verify: upsample -> downsample loopback
    np.random.seed(42)
    n_sym = 20
    x = np.random.randn(n_sym, NFFT + NCP) * 0.1
    x_up = np.zeros((n_sym, SYM_IN), dtype=np.float64)
    for s in range(n_sym):
        row_up = np.zeros(SYM_IN)
        row_up[::SPS] = x[s]
        x_up[s] = np.convolve(row_up, h, mode='full')[:SYM_IN]

    y = x_up @ T
    x_nocp = x[:, NCP:]
    corr_seq = np.abs(np.correlate(y[0], x_nocp[0], mode='full'))
    d = np.argmax(corr_seq) - (NFFT - 1)
    if d >= 0:
        ya, xa = y[:, d:], x_nocp[:, :NFFT - d]
    else:
        ya, xa = y[:, :NFFT + d], x_nocp[:, -d:]
    correlation = np.corrcoef(ya.flatten(), xa.flatten())[0, 1]
    print(f"    Verify: loopback delay={d}, corr={correlation:.6f}  {'PASS' if correlation > 0.99 else 'FAIL'}")


# ============================================================
# [5] ZC synchronization reference (RX)
# ============================================================
def gen_zc_ref(output_dir):
    """Generate ZC reference sequence: zero-interpolated + RRC-filtered versions.

    TX sends RRC-filtered ZC; RX sync kernel uses RRC-filtered reference
    for cross-correlation (ZC autocorrelation is a delta function).
    """
    print(f"\n[5] ZC reference [{ZC_REF_LEN}]")

    N = 256
    primes = [p for p in range(N, 1, -1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]
    N_zc = primes[0] if primes else N - 1
    n = np.arange(N_zc, dtype=np.float64)
    zc = np.exp(-1j * np.pi * ZC_U * n * (n + 1) / N_zc).astype(np.complex64)

    # Zero-interpolated upsampling: [256] -> [1024]
    ref = np.zeros(ZC_REF_LEN, dtype=np.complex64)
    seq = np.zeros(N, dtype=np.complex64)
    seq[:N_zc] = zc
    ref[::SPS] = seq[:N]
    ref /= np.sqrt(np.sum(np.abs(ref) ** 2))

    ref.real.astype(np.float16).tofile(os.path.join(output_dir, 'zc_ref_real.bin'))
    ref.imag.astype(np.float16).tofile(os.path.join(output_dir, 'zc_ref_imag.bin'))
    print(f"    N_zc={N_zc}, u={ZC_U}, energy={np.sum(np.abs(ref)**2):.6f}")

    # RRC-filtered version (used by fine_sync kernel)
    h = rrc_filter(BETA, SPAN, SPS)
    up_rrc = np.zeros(ZC_REF_LEN, dtype=np.complex64)
    up_rrc[::SPS] = seq[:N]
    ref_rrc_r = np.convolve(up_rrc.real, h, 'full')[:ZC_REF_LEN]
    ref_rrc_i = np.convolve(up_rrc.imag, h, 'full')[:ZC_REF_LEN]
    ref_rrc_r.astype(np.float16).tofile(os.path.join(output_dir, 'zc_ref_rrc_real.bin'))
    ref_rrc_i.astype(np.float16).tofile(os.path.join(output_dir, 'zc_ref_rrc_imag.bin'))
    print(f"    + RRC-filtered version (energy={np.sum(ref_rrc_r**2 + ref_rrc_i**2):.2f})")


# ============================================================
# [6] LS channel estimation matrix (RX)
# ============================================================
def gen_ls_matrix(output_dir):
    """LS interpolation matrix in interleaved real format.

    Maps pilot observations [batch, 2*N_PILOT] -> channel estimates [batch, 2*NFFT].
    Uses linear interpolation for non-uniform pilot spacing.

    Input layout:  [y_r0, y_i0, y_r1, y_i1, ..., y_r15, y_i15]  shape [batch, 32]
    Output layout: [h_r(256), h_i(256)]                           shape [batch, 512]
    """
    print(f"\n[6] LS matrix [{2*N_PILOT}, {2*NFFT}]")

    SCALE_l = 2.0 / np.sqrt(42)
    OFFSET_l = -7.0 / np.sqrt(42)
    LEVELS_l = np.arange(8) * SCALE_l + OFFSET_l
    pilot_val = np.max(np.abs(LEVELS_l)) * (1 + 1j)
    inv_pilot = 1.0 / pilot_val
    pilot_pos = np.array(PILOT_IDX)

    # Complex interpolation matrix L[k, p]
    L = np.zeros((NFFT, N_PILOT), dtype=np.complex128)
    for k in range(NFFT):
        if k <= pilot_pos[0]:
            L[k, 0] = inv_pilot
        elif k >= pilot_pos[-1]:
            L[k, -1] = inv_pilot
        else:
            for p in range(N_PILOT - 1):
                if pilot_pos[p] <= k <= pilot_pos[p + 1]:
                    w = (k - pilot_pos[p]) / (pilot_pos[p + 1] - pilot_pos[p])
                    L[k, p]     = (1 - w) * inv_pilot
                    L[k, p + 1] = w * inv_pilot
                    break

    # Convert to interleaved real format for Matmul
    ls_real = np.zeros((2 * N_PILOT, 2 * NFFT), dtype=np.float32)
    for p in range(N_PILOT):
        for k in range(NFFT):
            ls_real[2*p,   k]        =  L[k, p].real   # y_r_p -> h_r_k
            ls_real[2*p+1, k]        = -L[k, p].imag   # y_i_p -> h_r_k
            ls_real[2*p,   NFFT + k] =  L[k, p].imag   # y_r_p -> h_i_k
            ls_real[2*p+1, NFFT + k] =  L[k, p].real   # y_i_p -> h_i_k

    ls_real.astype(np.float16).tofile(os.path.join(output_dir, 'ls_matrix.bin'))

    # Verify with flat channel
    H_true = (0.8 + 0.2j) * np.ones(NFFT)
    rx_pilot = H_true[PILOT_IDX] * pilot_val
    p_il = np.zeros(2 * N_PILOT, dtype=np.float32)
    for i in range(N_PILOT):
        p_il[2*i]     = rx_pilot[i].real
        p_il[2*i + 1] = rx_pilot[i].imag
    h_ri = p_il @ ls_real
    h_est = h_ri[:NFFT] + 1j * h_ri[NFFT:]
    err = np.abs(h_est - H_true).max()
    print(f"    pilot_val={pilot_val:.4f}")
    print(f"    Verify: flat channel max_err={err:.2e}  {'PASS' if err < 0.01 else 'FAIL'}")


# ============================================================
# [7] FIR anti-aliasing Toeplitz matrix (optional)
# ============================================================
def gen_fir_toeplitz(output_dir):
    """FIR anti-aliasing filter Toeplitz matrix [1088, 1088]."""
    try:
        from scipy.signal import firwin
    except ImportError:
        print(f"\n[7] FIR Toeplitz skipped (scipy not installed)")
        return

    print(f"\n[7] FIR anti-aliasing Toeplitz matrix")
    CHUNK = 1088
    cutoff = (1.0 / SPS) * (1 + BETA)
    h = firwin(128, cutoff)
    L = len(h)
    T = np.zeros((CHUNK, CHUNK), dtype=np.float64)
    for i in range(CHUNK):
        for j in range(CHUNK):
            if 0 <= j - i < L:
                T[i, j] = h[j - i]
    T16 = T.astype(np.float16)
    T16.tofile(os.path.join(output_dir, 'fir_aa_toeplitz.bin'))
    print(f"    FIR: {L} taps, cutoff={cutoff:.3f}")
    print(f"    T [{CHUNK}, {CHUNK}] -> fir_aa_toeplitz.bin ({T16.nbytes/1024:.0f} KB)")


# ============================================================
# [8] LDPC matrices (PEG construction)
# ============================================================
def _peg_find_best_check(H, var_j, n_checks, check_degrees):
    """BFS-based check node selection to maximize local girth."""
    connected_checks = set(np.where(H[:, var_j] == 1)[0])
    visited_checks = set()
    visited_vars = {var_j}
    current_checks = set()
    for c in connected_checks:
        current_checks.add(c)
        visited_checks.add(c)
    last_layer_checks = set()
    while current_checks:
        last_layer_checks = current_checks
        next_vars = set()
        for c in current_checks:
            for v in np.where(H[c, :] == 1)[0]:
                if v not in visited_vars:
                    visited_vars.add(v)
                    next_vars.add(v)
        next_checks = set()
        for v in next_vars:
            for c in np.where(H[:, v] == 1)[0]:
                if c not in visited_checks:
                    visited_checks.add(c)
                    next_checks.add(c)
        current_checks = next_checks
    unvisited = set(range(n_checks)) - visited_checks
    if unvisited:
        candidates = list(unvisited)
    else:
        candidates = list(last_layer_checks - connected_checks)
        if not candidates:
            candidates = [c for c in range(n_checks) if c not in connected_checks]
        if not candidates:
            candidates = list(range(n_checks))
    min_deg = min(check_degrees[c] for c in candidates)
    best = [c for c in candidates if check_degrees[c] == min_deg]
    return best[np.random.randint(len(best))]


def peg_construct(n_checks, n_vars, col_weight):
    """Construct LDPC parity check matrix using Progressive Edge Growth."""
    H = np.zeros((n_checks, n_vars), dtype=np.uint8)
    check_degrees = np.zeros(n_checks, dtype=np.int32)
    for j in range(n_vars):
        if j % 100 == 0:
            print(f"      variable node {j}/{n_vars}...")
        for k in range(col_weight):
            if k == 0:
                min_deg = np.min(check_degrees)
                candidates = np.where(check_degrees == min_deg)[0]
                chosen = candidates[np.random.randint(len(candidates))]
            else:
                chosen = _peg_find_best_check(H, j, n_checks, check_degrees)
            H[chosen, j] = 1
            check_degrees[chosen] += 1
    return H


def gaussian_elimination_gf2(H):
    """Compute systematic generator matrix G from H over GF(2)."""
    m, n = H.shape
    k = n - m
    H_work = H.astype(np.int32).copy()
    col_perm = np.arange(n)
    for i in range(m):
        pivot_found = False
        for j in range(i, n):
            if H_work[i, j] == 1:
                pivot_found = True
                if j != i + k:
                    H_work[:, [j, i + k]] = H_work[:, [i + k, j]]
                    col_perm[[j, i + k]] = col_perm[[i + k, j]]
                break
        if not pivot_found:
            for ii in range(i + 1, m):
                if np.any(H_work[ii, i + k:] == 1):
                    jj = i + k + np.where(H_work[ii, i + k:] == 1)[0][0]
                    H_work[[i, ii], :] = H_work[[ii, i], :]
                    if jj != i + k:
                        H_work[:, [jj, i + k]] = H_work[:, [i + k, jj]]
                        col_perm[[jj, i + k]] = col_perm[[i + k, jj]]
                    pivot_found = True
                    break
        if not pivot_found:
            continue
        for ii in range(m):
            if ii != i and H_work[ii, i + k] == 1:
                H_work[ii] ^= H_work[i]
    P = H_work[:, :k]
    G_perm = np.zeros((k, n), dtype=np.uint8)
    G_perm[:, :k] = np.eye(k, dtype=np.uint8)
    G_perm[:, k:] = P.T % 2
    inv_perm = np.argsort(col_perm)
    return G_perm[:, inv_perm]


def gf2_inv(Mat):
    """Compute matrix inverse over GF(2) via Gauss-Jordan."""
    n = Mat.shape[0]
    A = np.hstack([Mat.copy().astype(np.int32), np.eye(n, dtype=np.int32)])
    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if A[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            return None
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
        for row in range(n):
            if row != col and A[row, col] == 1:
                A[row] = (A[row] + A[col]) % 2
    return A[:, n:]


def gf2_rank(Mat):
    """Compute matrix rank over GF(2)."""
    m, n = Mat.shape
    A = Mat.astype(np.int32).copy()
    rank = 0
    for col in range(n):
        pivot = -1
        for row in range(rank, m):
            if A[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]
        for row in range(m):
            if row != rank and A[row, col] == 1:
                A[row] ^= A[rank]
        rank += 1
    return rank


def gen_ldpc_matrices(output_dir):
    """Construct LDPC code via PEG, generate H^T, G, G_left_inv."""
    print(f"\n[8] LDPC matrices (PEG, codeword={LDPC_K}, parity={LDPC_N})")
    np.random.seed(LDPC_SEED)

    # PEG construction
    print(f"    Constructing H [{LDPC_N}, {LDPC_K}]...")
    H = peg_construct(LDPC_N, LDPC_K, LDPC_COL_WEIGHT)
    col_w = np.sum(H, axis=0)
    row_w = np.sum(H, axis=1)
    print(f"    Column weight: {col_w.min()}-{col_w.max()}, row weight: {row_w.min()}-{row_w.max()}")

    rank = gf2_rank(H)
    print(f"    Rank(H) = {rank} (expected {LDPC_N})  {'PASS' if rank == LDPC_N else 'FAIL'}")
    if rank < LDPC_N:
        print(f"    FAIL: rank deficient, try different LDPC_SEED")
        return False

    # Generator matrix
    print(f"    Constructing G [{LDPC_K - LDPC_N}, {LDPC_K}]...")
    G = gaussian_elimination_gf2(H)
    check = (H @ G.T) % 2
    print(f"    H * G^T = 0: {'PASS' if np.sum(check) == 0 else 'FAIL'}")

    # Left inverse of G[:, :N] for information bit recovery
    print(f"    Constructing G_left_inv [{LDPC_N}, {LDPC_N}]...")
    G_left = G[:, :LDPC_N].astype(np.int32)
    G_left_inv = gf2_inv(G_left)
    if G_left_inv is None:
        print(f"    FAIL: G[:,:N] is singular")
        return False
    check_inv = (G_left_inv @ G_left) % 2
    print(f"    G_left_inv * G_left = I: {'PASS' if np.array_equal(check_inv, np.eye(LDPC_N, dtype=np.int32)) else 'FAIL'}")

    # Save matrices
    H_T = H.T.astype(np.uint8)
    H_T.tofile(os.path.join(output_dir, 'matrix_H.bin'))
    G.astype(np.uint8).tofile(os.path.join(output_dir, 'matrix_G.bin'))
    G_left_inv.astype(np.int8).tofile(os.path.join(output_dir, 'matrix_G_left_inv.bin'))
    print(f"    matrix_H.bin           H^T {H_T.shape} uint8")
    print(f"    matrix_G.bin           G   {G.shape} uint8")
    print(f"    matrix_G_left_inv.bin  ({LDPC_N},{LDPC_N}) int8")

    # Verify with test codewords
    info_bits = np.random.randint(0, 2, size=(LDPC_M, LDPC_K - LDPC_N)).astype(np.int32)
    codewords = ((info_bits @ G.astype(np.int32)) % 2).astype(np.int8)
    syn = (codewords.astype(np.int32) @ H_T.astype(np.int32)) % 2
    valid = np.sum(np.sum(syn, axis=1) == 0)
    print(f"    Codeword verify: {valid}/{LDPC_M} valid")

    # Save test vectors for encoder verification
    info_bits.astype(np.int8).tofile(os.path.join(output_dir, 'x1_gm.bin'))
    codewords.tofile(os.path.join(output_dir, 'golden_enc_output.bin'))

    # Golden decoder output
    out_dir = os.path.join(os.path.dirname(output_dir), 'output')
    os.makedirs(out_dir, exist_ok=True)
    codewords.tofile(os.path.join(out_dir, 'golden.bin'))
    golden_info = ((codewords[:, :LDPC_N].astype(np.int32) @ G_left_inv) % 2).astype(np.int8)
    golden_info.tofile(os.path.join(out_dir, 'golden_info.bin'))
    print(f"    + test vectors: x1_gm.bin, golden_enc_output.bin, output/golden*.bin")

    return True


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Generate OFDM ISAC precomputed matrices')
    parser.add_argument('--output_dir', default='./input')
    parser.add_argument('--skip-ldpc', action='store_true', help='Skip LDPC (slow PEG construction)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  OFDM ISAC Matrix Generator")
    print("=" * 60)
    print(f"  NFFT={NFFT}, NCP={NCP}, SPS={SPS}, beta={BETA}")
    print(f"  data={N_DATA}, pilot={N_PILOT}, guard={N_GUARD}")
    print(f"  Output: {args.output_dir}/")
    print("=" * 60)

    gen_ifft_matrix(args.output_dir)
    gen_fft_matrix(args.output_dir)
    gen_rrc_toeplitz(args.output_dir)
    gen_rrc_down_matrix(args.output_dir)
    gen_zc_ref(args.output_dir)
    gen_ls_matrix(args.output_dir)
    gen_fir_toeplitz(args.output_dir)

    if not args.skip_ldpc:
        gen_ldpc_matrices(args.output_dir)
    else:
        print(f"\n[8] LDPC skipped (--skip-ldpc)")

    # File listing
    print(f"\n{'=' * 60}")
    print("Generated files:")
    print(f"{'=' * 60}")
    total = 0
    for f in sorted(os.listdir(args.output_dir)):
        path = os.path.join(args.output_dir, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            total += size
            print(f"  {f:35s} {size/1024:8.1f} KB")
    print(f"  {'Total':35s} {total/1024:8.1f} KB")
    print()


if __name__ == '__main__':
    main()