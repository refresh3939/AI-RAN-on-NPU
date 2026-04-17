#!/usr/bin/env python3
"""
tx_usrp_npu.py — OFDM ISAC Transmitter (Ascend 310B1 + USRP X300)

Complete over-the-air OFDM transmitter with all baseband DSP on Ascend NPU.
No PyTorch dependency — pure AscendC custom operators via ACL pybind.

Signal path:
  info bits → LDPC encode → bit pack → QAM64 → OFDM IFFT → RRC upsample
  → sample interleave → USRP sc16

Expected directory layout (relative to this script):
  ./build/                — compiled .so module
  ./input/                — binary constant files
  ./tx_bits.npy           — output (TX reference bits for RX BER verification)

Usage:
  python3 tx_usrp_npu.py --gain 0 --count 100
  python3 tx_usrp_npu.py --gain 0            # stream until Ctrl+C
  python3 tx_usrp_npu.py --no-usrp           # generate signal only

Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
Target: Ascend 310B1 (Orange Pi AI Pro)
SPDX-License-Identifier: MIT
"""
# UHD must be imported before any ACL-related modules
import uhd

import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'

import sys
import numpy as np
import time
import argparse

# ============================================================
# Paths (resolved relative to this script's directory)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR  = os.path.join(SCRIPT_DIR, 'build')
INPUT_DIR  = os.path.join(SCRIPT_DIR, 'input')

# ============================================================
# System parameters
# ============================================================
N_FFT       = 256
NCP         = 16
SPS         = 4
N_SYM       = 1192
SYM_TD      = N_FFT + NCP
SYM_UP      = SYM_TD * SPS
K_DATA      = 220
ZC_N        = 256
LDPC_K      = 256
LDPC_N      = 512
FRAME_LEN   = N_SYM * SYM_UP

USRP_ADDR   = "addr=192.168.20.2"
TX_FREQ     = 2.45e9
TX_RATE     = 5e6
TX_GAIN     = 0
SEED        = 42
GUARD_LEN   = 5000

W    = 78
LINE = '-' * W


def load_bin(name, dtype):
    """Load binary constant file from input directory."""
    return np.fromfile(os.path.join(INPUT_DIR, name), dtype=dtype)


def gen_zc_sequence(N=256, u=1):
    """Generate raw Zadoff-Chu sync sequence (not upsampled, not normalized)."""
    primes = [p for p in range(N, 1, -1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]
    N_zc = primes[0] if primes else N - 1
    n = np.arange(N_zc, dtype=np.float64)
    zc = np.exp(-1j * np.pi * u * n * (n + 1) / N_zc).astype(np.complex64)
    seq = np.zeros(N, dtype=np.complex64)
    seq[:N_zc] = zc
    return seq


def gen_random_bits(seed=42):
    """Generate TX information bits."""
    np.random.seed(seed)
    total_coded_bits = N_SYM * K_DATA * 6
    total_info_bits  = total_coded_bits // 2
    n_codewords = total_info_bits // LDPC_K
    bits = np.random.randint(0, 2, size=n_codewords * LDPC_K).astype(np.int8)
    return bits, n_codewords


def init_tx_chain():
    """Initialize NPU TX chain: load constants, allocate GM buffers."""
    sys.path.insert(0, BUILD_DIR)
    import ascend_baseband_tx_chain as tx_npu

    G = load_bin('x2_gm.bin', np.int8).reshape(LDPC_K, LDPC_N)
    ifft_cos = load_bin('matG_ifft_real.bin', np.float16).reshape(SYM_TD, SYM_TD)
    ifft_sin = load_bin('matG_ifft_imag.bin', np.float16).reshape(SYM_TD, SYM_TD)
    rrc = np.concatenate([
        load_bin(f'rrc_toeplitz_h{p}.bin', np.float16).reshape(SYM_TD, SYM_TD)
        for p in range(SPS)
    ]).copy()

    tx_npu.init_resources(
        G.copy(),
        ifft_cos.view(np.uint16).copy(),
        ifft_sin.view(np.uint16).copy(),
        rrc.view(np.uint16).copy())
    return tx_npu


# Pre-loaded constants (pilot bias and ZC sequence, computed once per session)
_pilot_r = None
_pilot_i = None
_zc_r    = None
_zc_i    = None


def preload_constants():
    """Pre-load pilot bias and ZC sequence (invariant across frames)."""
    global _pilot_r, _pilot_i, _zc_r, _zc_i
    _pilot_r = load_bin('pilot_bias_real.bin', np.float16).view(np.uint16).copy()
    _pilot_i = load_bin('pilot_bias_imag.bin', np.float16).view(np.uint16).copy()
    zc = gen_zc_sequence(ZC_N, 1)
    _zc_r = zc.real.astype(np.float16).view(np.uint16).copy()
    _zc_i = zc.imag.astype(np.float16).view(np.uint16).copy()


def tx_process(tx_npu, bits, n_codewords):
    """Generate one frame of baseband signal via the NPU TX chain."""
    info_bits = bits[:n_codewords * LDPC_K].reshape(n_codewords, LDPC_K)
    result = tx_npu.run_tx(info_bits, _pilot_r, _pilot_i, _zc_r, _zc_i)

    bb_r = result[0].view(np.float16).astype(np.float32)
    bb_i = result[1].view(np.float16).astype(np.float32)
    bb = (bb_r + 1j * bb_i).astype(np.complex64)

    # Normalize to peak = 1.0
    mx = np.max(np.abs(bb))
    if mx > 1e-10:
        bb = bb / mx
    return bb


def main():
    parser = argparse.ArgumentParser(description='OFDM ISAC TX (Ascend 310B1 + USRP X300)')
    parser.add_argument('--gain',  type=float, default=TX_GAIN)
    parser.add_argument('--freq',  type=float, default=TX_FREQ)
    parser.add_argument('--rate',  type=float, default=TX_RATE)
    parser.add_argument('--count', type=int, default=0, help='0 = run forever')
    parser.add_argument('--guard', type=int, default=GUARD_LEN)
    parser.add_argument('--no-usrp', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print(f"\n  {'=' * W}")
    print(f"  {'OFDM ISAC TX — Ascend 310B1 + USRP X300':^{W}}")
    print(f"  {'=' * W}")
    print(f"  Rate: {args.rate/1e6:.1f} MHz  Freq: {args.freq/1e9:.3f} GHz  "
          f"Gain: {args.gain:.0f} dB  Guard: {args.guard} samples")
    print(f"  {LINE}")

    # Initialize
    print(f"  [Init] Loading TX chain...")
    tx_npu = init_tx_chain()
    preload_constants()

    # Generate random bits (fixed seed for RX BER verification)
    bits, n_cw = gen_random_bits(seed=SEED)

    print(f"  [Warmup]...")
    tx_npu.warmup(bits[:n_cw * LDPC_K].reshape(n_cw, LDPC_K),
                  _pilot_r, _pilot_i, _zc_r, _zc_i)
    print(f"  [Init] Warmup done")

    # Generate one frame
    t0 = time.perf_counter()
    bb = tx_process(tx_npu, bits, n_cw)
    dt_gen = (time.perf_counter() - t0) * 1000

    rms = np.sqrt(np.mean(bb.real**2 + bb.imag**2))
    pk  = np.sqrt(np.max(bb.real**2 + bb.imag**2))
    papr = 10 * np.log10(pk**2 / (rms**2 + 1e-20))

    # Save TX bits for RX BER verification (script directory)
    save_path = os.path.join(SCRIPT_DIR, 'tx_bits.npy')
    np.save(save_path, bits)

    # Build transmit frame: [guard | baseband | guard]
    guard = np.zeros(args.guard, dtype=np.complex64)
    frame = np.concatenate([guard, bb.astype(np.complex64), guard])
    frame_ms = len(frame) / args.rate * 1000

    print(f"  [TX]   Generated {len(bb)} samples ({dt_gen:.0f}ms)  "
          f"RMS={rms:.4f}  Peak={pk:.4f}  PAPR={papr:.1f}dB")
    print(f"  [TX]   Frame: {len(bb)} + 2×{args.guard} guard = {len(frame)} "
          f"({frame_ms:.0f}ms)")
    print(f"  [Save] tx_bits.npy → {save_path}")

    if args.no_usrp:
        print(f"\n  [USRP] Skipped (--no-usrp)")
        tx_npu.cleanup()
        return

    # USRP
    print(f"\n  [USRP] Connecting to {USRP_ADDR}...")
    usrp = uhd.usrp.MultiUSRP(USRP_ADDR)
    usrp.set_tx_rate(args.rate)
    usrp.set_tx_freq(uhd.types.TuneRequest(args.freq))
    usrp.set_tx_gain(args.gain)
    usrp.set_tx_antenna("TX/RX")
    streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    print(f"  [USRP] {usrp.get_mboard_name()} ready")

    # Streaming loop
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst   = False

    print(f"\n  {'#':>5s}  {'Send':>6s}  {'FPS':>5s}  {'Elapsed':>8s}")
    print(f"  {LINE}")

    n_sent = 0
    t_start = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            streamer.send(frame, md)
            dt_send = (time.perf_counter() - t0) * 1000
            md.start_of_burst = False

            n_sent += 1
            elapsed = time.perf_counter() - t_start
            fps = n_sent / elapsed if elapsed > 0 else 0

            if n_sent <= 3 or n_sent % 50 == 0:
                # Per-module timing from C++
                t = tx_npu.get_timing()
                print(f"  {n_sent:5d}  {dt_send:5.0f}ms  {fps:5.1f}  {elapsed:7.1f}s")
                print(f"         LDPC:{t['ldpc']:5.1f}  QAM:{t['bit_pack']+t['qam']:5.1f}  "
                      f"IFFT:{t['ifft']:4.1f}  Post:{t['postproc']:3.1f}  "
                      f"RrcPrep:{t['rrc_prep']:4.1f}  RRC:{t['rrc']:4.1f}  "
                      f"Intlv:{t['interleave']:4.1f}  Total:{t['total']:5.1f}ms")

            if 0 < args.count <= n_sent:
                break

    except KeyboardInterrupt:
        print(f"\n  [Stop] Ctrl+C")

    md.end_of_burst = True
    streamer.send(frame, md)

    elapsed = time.perf_counter() - t_start
    fps = n_sent / elapsed if elapsed > 0 else 0

    print(f"\n  {'=' * W}")
    print(f"  Frames sent: {n_sent}  |  Elapsed: {elapsed:.1f}s  |  FPS: {fps:.1f}")
    print(f"  {'=' * W}\n")
    tx_npu.cleanup()


if __name__ == '__main__':
    main()
