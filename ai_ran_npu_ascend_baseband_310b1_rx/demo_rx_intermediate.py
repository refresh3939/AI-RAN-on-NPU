#!/usr/bin/env python3
"""
demo_rx_intermediate.py — RX Pipeline Intermediate Output Extraction

Runs the full NPU RX chain on a captured signal and extracts intermediate
data from each processing stage for debugging and visualization:
  - Time-domain waveform, constellation, channel response, spectrum
  - SNR, EVM, BER analysis

Usage:
  python3 demo_rx_intermediate.py                        # default capture
  python3 demo_rx_intermediate.py --capture data.npy     # custom capture
  python3 demo_rx_intermediate.py --save-plots           # save figures
  python3 demo_rx_intermediate.py --dump-all             # export all data

Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
Target: Ascend 310B1 (Orange Pi AI Pro)
SPDX-License-Identifier: MIT
"""
import os
import sys
import numpy as np
import argparse
import time

# ============================================================
# System parameters
# ============================================================
N_FFT       = 256
NCP         = 16
SPS         = 4
N_SYM       = 1192
SYM_TD      = N_FFT + NCP       # 272
SYM_UP      = SYM_TD * SPS      # 1088
K_DATA      = 220
K_DATA_PAD  = 224                # 32B-aligned
K_PILOT     = 16
ZC_N        = 256
ZC_UP       = ZC_N * SPS        # 1024
LDPC_K      = 256
FRAME_LEN   = N_SYM * SYM_UP
COMP_LEN    = ZC_UP + FRAME_LEN

RX_DIR    = os.path.expanduser('~/aclsystem')
INPUT_DIR = os.path.join(RX_DIR, 'input')


def load_bin(name, dtype):
    return np.fromfile(os.path.join(INPUT_DIR, name), dtype=dtype)


def init_rx():
    """Initialize NPU RX chain."""
    sys.path.insert(0, os.path.join(RX_DIR, 'build'))
    import ascend_baseband_rx_chain as rx

    dc = np.exp(1j * 2 * np.pi * np.arange(N_FFT) * 8 / N_FFT).astype(np.complex64)
    rx.init_resources(
        dc.real.astype(np.float16).view(np.uint16).copy(),
        dc.imag.astype(np.float16).view(np.uint16).copy(),
        load_bin('pilot_subcarrier_idx.bin', np.int32).copy(),
        load_bin('data_subcarrier_idx.bin', np.int32).copy(),
        load_bin('zc_ref_rrc_real.bin', np.float16).view(np.uint16).copy(),
        load_bin('zc_ref_rrc_imag.bin', np.float16).view(np.uint16).copy(),
        load_bin('rrc_down_T.bin', np.float16).view(np.uint16).copy(),
        load_bin('matG_fft_real.bin', np.float16).view(np.uint16).copy(),
        load_bin('matG_fft_imag.bin', np.float16).view(np.uint16).copy(),
        load_bin('ls_matrix.bin', np.float16).view(np.uint16).copy(),
        load_bin('matrix_H.bin', np.uint8).astype(np.int8).copy(),
        load_bin('matrix_G_left_inv.bin', np.int8).copy())
    rx.warmup()
    return rx


def dump_half(rx, name, n_elems):
    """Download GM buffer as float32 (stored as float16 on device)."""
    return rx.dump_gm(name, n_elems).view(np.float16).astype(np.float32)


def dump_complex(rx, name_r, name_i, n_elems):
    """Download complex GM buffer (separate R/I halves)."""
    return dump_half(rx, name_r, n_elems) + 1j * dump_half(rx, name_i, n_elems)


def complex64_to_sc16(buf):
    """Convert complex64 numpy array to sc16 interleaved int16."""
    r = (buf.real * 32768.0).clip(-32768, 32767).astype(np.int16)
    i = (buf.imag * 32768.0).clip(-32768, 32767).astype(np.int16)
    sc16 = np.empty(len(buf) * 2, dtype=np.int16)
    sc16[0::2] = r
    sc16[1::2] = i
    return sc16


def main():
    parser = argparse.ArgumentParser(description='RX Pipeline Intermediate Output Extraction')
    parser.add_argument('--capture', default=os.path.join(RX_DIR, 'usrp_cap.npy'))
    parser.add_argument('--txbits', default=os.path.expanduser('~/system/tx_bits.npy'))
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--dump-all', action='store_true')
    args = parser.parse_args()

    SEP = '=' * 60
    print(f"\n{SEP}")
    print(f"  RX Pipeline Intermediate Output Extraction")
    print(f"{SEP}")

    # ---- [1] Load capture ----
    print(f"\n[1] Load capture: {args.capture}")
    buf = np.load(args.capture)
    print(f"    shape={buf.shape}  dtype={buf.dtype}  "
          f"duration={len(buf)/5e6*1000:.0f}ms  RMS={np.sqrt(np.mean(np.abs(buf)**2)):.4f}")

    # ---- [2] Init NPU ----
    print(f"\n[2] Init NPU RX chain...")
    rx = init_rx()

    # ---- [3] H2D (complex64 → sc16 → f16 → GM) ----
    print(f"\n[3] H2D upload...")
    sc16 = complex64_to_sc16(buf)
    rx.run_h2d(sc16)

    # ---- [4] Sync ----
    print(f"\n[4] Synchronization...")
    t0 = time.perf_counter()
    max_search = max(0, len(buf) - ZC_UP - FRAME_LEN)
    pos, cfo = rx.run_sync(0, max_search)
    dt_sync = (time.perf_counter() - t0) * 1000
    print(f"    pos={pos}  CFO={cfo:.1f}Hz  time={dt_sync:.1f}ms")

    if pos < 0:
        print("    [FAIL] Sync failed, aborting.")
        rx.cleanup()
        return

    # ---- [5] Decode ----
    print(f"\n[5] Decode pipeline...")
    t0 = time.perf_counter()
    info_bits = rx.run_decode(pos, cfo, LDPC_K)
    dt_dec = (time.perf_counter() - t0) * 1000
    print(f"    time={dt_dec:.1f}ms")

    # ---- BER ----
    ber = -1.0
    if os.path.exists(args.txbits):
        tx_bits = np.load(args.txbits).astype(np.int8)
        rx_bits = (info_bits.flatten() & 1).astype(np.int8)
        n = min(len(tx_bits), len(rx_bits))
        ber = float(np.mean(tx_bits[:n] != rx_bits[:n])) if n > 0 else -1.0
        print(f"    BER={ber:.6f} ({n} bits)")

    # ---- [6] Extract intermediate outputs ----
    print(f"\n[6] Extract intermediate outputs...")

    pilot_sc = load_bin('pilot_subcarrier_idx.bin', np.int32)
    data_sc  = load_bin('data_subcarrier_idx.bin', np.int32)

    # (a) CFO compensated signal
    comp = dump_complex(rx, 'comp_r', 'comp_i', COMP_LEN)
    print(f"    comp:  [{COMP_LEN}]  RMS={np.sqrt(np.mean(np.abs(comp)**2)):.4f}")

    # (b) RRC filtered + downsampled (stride=272, data at offset 16)
    rrc_raw = dump_complex(rx, 'rrc_r', 'rrc_i', N_SYM * SYM_TD).reshape(N_SYM, SYM_TD)
    rrc = rrc_raw[:, NCP:]  # skip CP region (zeros), keep N_FFT samples
    print(f"    rrc:   [{N_SYM}, {N_FFT}]  RMS={np.sqrt(np.mean(np.abs(rrc)**2)):.4f}")

    # (c) Frequency domain (after FFT + delay compensation)
    freq = dump_complex(rx, 'freq_r', 'freq_i', N_SYM * N_FFT).reshape(N_SYM, N_FFT)
    print(f"    freq:  [{N_SYM}, {N_FFT}]")

    # (d) Pilot observations (interleaved R/I)
    pilot_raw = dump_half(rx, 'pilot', N_SYM * K_PILOT * 2).reshape(N_SYM, K_PILOT * 2)
    pilot = pilot_raw[:, 0::2] + 1j * pilot_raw[:, 1::2]
    print(f"    pilot: [{N_SYM}, {K_PILOT}]")

    # (e) Equalized frequency domain
    eq = dump_complex(rx, 'eq_r', 'eq_i', N_SYM * N_FFT).reshape(N_SYM, N_FFT)
    print(f"    eq:    [{N_SYM}, {N_FFT}]")

    # (f) QAM constellation (data subcarriers, padded to 224)
    qam_pad = dump_complex(rx, 'data_r', 'data_i', N_SYM * K_DATA_PAD).reshape(N_SYM, K_DATA_PAD)
    qam = qam_pad[:, :K_DATA]  # trim padding
    print(f"    qam:   [{N_SYM}, {K_DATA}]")

    # ---- [7] Analysis ----
    print(f"\n[7] Analysis...")

    # Channel from equalized / raw ratio at data subcarriers
    H_from_eq = eq[:, data_sc]
    raw_data = freq[:, data_sc]
    H_est = np.where(np.abs(qam) > 1e-6, raw_data / qam, 0)

    # EVM (error vector magnitude)
    qam_norm = 1.0 / np.sqrt(42.0)
    ideal = np.round(qam / qam_norm) * qam_norm
    evm_rms = np.sqrt(np.mean(np.abs(qam - ideal)**2))
    sig_rms = np.sqrt(np.mean(np.abs(qam)**2))
    evm_db = 20 * np.log10(evm_rms / (sig_rms + 1e-20) + 1e-20)
    snr_evm = -evm_db
    print(f"    EVM:       {evm_db:.1f} dB")
    print(f"    SNR (EVM): {snr_evm:.1f} dB")

    # SNR from pilot variance
    H_pilot = pilot / (pilot[0:1, :] + 1e-20)  # normalize to first symbol
    H_mean = np.mean(H_pilot, axis=0, keepdims=True)
    noise_var = np.mean(np.abs(H_pilot - H_mean)**2, axis=0)
    sig_pow = np.abs(H_mean[0])**2
    snr_pilot = 10 * np.log10(sig_pow / (noise_var + 1e-20))
    print(f"    SNR (pilot): {np.mean(snr_pilot):.1f} dB "
          f"(min={np.min(snr_pilot):.1f}  max={np.max(snr_pilot):.1f})")

    # ---- [8] Plots ----
    if args.save_plots:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plot_dir = 'plots'
            os.makedirs(plot_dir, exist_ok=True)
            print(f"\n[8] Saving plots to {plot_dir}/")

            # 8a: Time domain
            fig, axes = plt.subplots(3, 1, figsize=(14, 8))
            t_ms = np.arange(len(buf)) / 5e6 * 1000
            axes[0].plot(t_ms[::4], np.abs(buf[::4]), linewidth=0.3)
            axes[0].axvline(pos / 5e6 * 1000, color='r', ls='--', label=f'sync pos={pos}')
            axes[0].set(ylabel='|signal|', title='Received Signal'); axes[0].legend()

            t_comp = np.arange(COMP_LEN) / 5e6 * 1000
            axes[1].plot(t_comp[::40], np.abs(comp[::40]), linewidth=0.3)
            axes[1].set(ylabel='|comp|', title=f'After CFO Compensation ({cfo:.0f}Hz)')

            axes[2].plot(np.abs(rrc[0]), linewidth=0.5)
            axes[2].set(ylabel='|rrc[0]|', title='RRC Downsampled — Symbol 0')
            plt.tight_layout(); plt.savefig(f'{plot_dir}/1_time_domain.png', dpi=150); plt.close()
            print(f"    1_time_domain.png")

            # 8b: Constellation (before/after EQ)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            freq_data = freq[:, data_sc]
            axes[0].scatter(freq_data.real.flat[::20], freq_data.imag.flat[::20], s=0.2, alpha=0.5)
            axes[0].set(title='Before Equalization', aspect='equal'); axes[0].grid(True, alpha=0.3)
            axes[1].scatter(qam.real.flat[::20], qam.imag.flat[::20], s=0.2, alpha=0.5, c='green')
            axes[1].set(title='After Equalization (64-QAM)', aspect='equal'); axes[1].grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(f'{plot_dir}/2_constellation.png', dpi=150); plt.close()
            print(f"    2_constellation.png")

            # 8c: Subcarrier power spectrum
            fig, ax = plt.subplots(figsize=(12, 4))
            eq_pwr = 10 * np.log10(np.mean(np.abs(eq)**2, axis=0) + 1e-20)
            ax.plot(eq_pwr, linewidth=0.8)
            ax.axhline(np.mean(eq_pwr[data_sc]), color='r', ls='--',
                       label=f'data mean={np.mean(eq_pwr[data_sc]):.1f}dB')
            for sc in pilot_sc: ax.axvline(sc, color='g', alpha=0.3, linewidth=0.5)
            ax.set(xlabel='Subcarrier', ylabel='Power (dB)', title='Equalized Subcarrier Power')
            ax.legend(); plt.tight_layout()
            plt.savefig(f'{plot_dir}/3_spectrum.png', dpi=150); plt.close()
            print(f"    3_spectrum.png")

            # 8d: EVM per symbol
            fig, ax = plt.subplots(figsize=(12, 3))
            evm_sym = 20 * np.log10(np.sqrt(np.mean(np.abs(qam - ideal)**2, axis=1))
                                    / (sig_rms + 1e-20) + 1e-20)
            ax.plot(evm_sym, linewidth=0.5)
            ax.axhline(evm_db, color='r', ls='--', alpha=0.5)
            ax.set(xlabel='OFDM Symbol', ylabel='EVM (dB)', title=f'EVM per Symbol (mean={evm_db:.1f}dB)')
            plt.tight_layout(); plt.savefig(f'{plot_dir}/4_evm.png', dpi=150); plt.close()
            print(f"    4_evm.png")

            print(f"    All plots saved!")
        except ImportError:
            print(f"\n[8] matplotlib not installed — pip install matplotlib")

    # ---- [9] Dump all ----
    if args.dump_all:
        dump_dir = 'dump'
        os.makedirs(dump_dir, exist_ok=True)
        print(f"\n[9] Export to {dump_dir}/")
        for name, data in [('comp', comp), ('rrc', rrc), ('freq', freq),
                           ('pilot', pilot), ('eq', eq), ('qam', qam),
                           ('info_bits', info_bits)]:
            np.save(f'{dump_dir}/{name}.npy', data)
            print(f"    {name}.npy  {data.shape}  {data.dtype}")

    # ---- Summary ----
    print(f"\n{SEP}")
    print(f"  Sync: pos={pos}  CFO={cfo:.1f}Hz  |  Decode: {dt_dec:.0f}ms  |  "
          f"BER: {ber:.6f}  |  EVM: {evm_db:.1f}dB  |  SNR: {snr_evm:.1f}dB")
    print(f"{SEP}\n")
    rx.cleanup()


if __name__ == '__main__':
    main()