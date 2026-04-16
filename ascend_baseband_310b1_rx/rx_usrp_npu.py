#!/usr/bin/env python3
"""
rx_usrp_npu.py — OFDM ISAC Receiver (Ascend 310B1 + USRP X300)

Complete over-the-air OFDM receiver with all baseband DSP on Ascend NPU.
No PyTorch dependency — pure AscendC custom operators via ACL pybind.

Signal path:
  USRP sc16 → NEON sc16→f16 → H2D DMA →
  Sync (3-phase) → CFO → RRC → FFT → EQ → QAM → LDPC → BER

Expected directory layout (relative to this script):
  ./build/                — compiled .so module
  ./input/                — binary constant files
  ./tx_bits.npy           — optional TX reference bits for BER check

Usage:
  python3 rx_usrp_npu.py --gain 10 --count 100
  python3 rx_usrp_npu.py --gain 10  # run until Ctrl+C

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
N_FFT           = 256
NCP             = 16
SPS             = 4
N_SYM           = 1192
SYM_TD          = N_FFT + NCP
SYM_UP          = SYM_TD * SPS
K_DATA          = 220
K_PILOT         = 16
ZC_N            = 256
ZC_UP           = ZC_N * SPS
LDPC_K          = 256
LDPC_N          = 512
FRAME_LEN       = N_SYM * SYM_UP
CAPTURE_SAMPLES = 2800000

USRP_ADDR       = "addr=192.168.10.3"
RX_FREQ         = 3e9
RX_RATE         = 5e6
RX_GAIN         = 20

W    = 78
LINE = '-' * W


def load_bin(name, dtype):
    """Load binary constant file from input directory."""
    return np.fromfile(os.path.join(INPUT_DIR, name), dtype=dtype)


def init_rx_chain():
    """Initialize NPU RX chain: load constants, allocate GM buffers."""
    sys.path.insert(0, BUILD_DIR)
    import ascend_baseband_rx_chain as rx_npu

    # Delay compensation vector
    dc = np.exp(1j * 2 * np.pi * np.arange(N_FFT) * 8 / N_FFT).astype(np.complex64)

    rx_npu.init_resources(
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

    return rx_npu


def rx_process(rx_npu, sig, tx_bits=None):
    """Process one captured frame through the full NPU RX chain.

    Args:
        rx_npu: pybind module (ascend_baseband_rx_chain)
        sig: int16 array, sc16 interleaved [R,I,R,I,...]
        tx_bits: reference bits for BER (optional)

    Returns:
        dict with decoded bits, BER, timing, etc.
    """
    result = {}
    pc = time.perf_counter

    # H2D: sc16 → f16 (ARM NEON) + DMA upload
    t0 = pc()
    rx_npu.run_h2d(sig)
    t_h2d = pc() - t0

    # Sync: energy scan → cross-correlation → CP-based CFO
    t0 = pc()
    search_len = min(len(sig) // 2 - ZC_UP - FRAME_LEN, CAPTURE_SAMPLES // 2)
    if search_len < 1000:
        search_len = len(sig) // 4

    try:
        fine_pos, frac_cfo = rx_npu.run_sync(0, search_len)
    except Exception as e:
        result['error'] = str(e)
        return result
    t_sync = pc() - t0
    result['fine_pos'] = fine_pos
    result['frac_cfo'] = frac_cfo

    # Decode: CFO → RRC → FFT → EQ → QAM → LDPC
    t0 = pc()
    try:
        info_bits = rx_npu.run_decode(fine_pos, frac_cfo, LDPC_K)
    except Exception as e:
        result['error'] = str(e)
        return result
    t_decode = pc() - t0
    result['bits'] = info_bits

    # Get per-module timing from C++
    result['timing'] = rx_npu.get_timing()

    # BER
    t0 = pc()
    if tx_bits is not None:
        rx_bits = (info_bits.flatten() & 1).astype(np.int8)
        n = min(len(tx_bits), len(rx_bits))
        result['ber'] = float(np.mean(tx_bits[:n] != rx_bits[:n])) if n > 0 else -1.0
    else:
        result['ber'] = -1.0
    t_ber = pc() - t0

    result['t_h2d_ms']    = t_h2d * 1000
    result['t_sync_ms']   = t_sync * 1000
    result['t_decode_ms'] = t_decode * 1000
    result['t_ber_ms']    = t_ber * 1000
    result['t_total_ms']  = (t_h2d + t_sync + t_decode + t_ber) * 1000
    return result


def signal_stats(sig_sc16):
    """Compute power (dBFS) and peak from sc16 buffer."""
    r = sig_sc16[0::2].astype(np.float32) / 32768.0
    i = sig_sc16[1::2].astype(np.float32) / 32768.0
    pwr = 10 * np.log10(np.mean(r * r + i * i) + 1e-20)
    pk = np.sqrt(np.max(r * r + i * i))
    return pwr, pk


def main():
    parser = argparse.ArgumentParser(description='OFDM ISAC RX (Ascend 310B1 + USRP X300)')
    parser.add_argument('--gain', type=float, default=RX_GAIN)
    parser.add_argument('--freq', type=float, default=RX_FREQ)
    parser.add_argument('--rate', type=float, default=RX_RATE)
    parser.add_argument('--count', type=int, default=0, help='0 = run forever')
    parser.add_argument('--capture-len', type=int, default=CAPTURE_SAMPLES)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    cap_ms = args.capture_len / args.rate * 1000
    print(f"\n  {'=' * W}")
    print(f"  {'OFDM ISAC RX — Ascend 310B1 + USRP X300':^{W}}")
    print(f"  {'=' * W}")
    print(f"  Rate: {args.rate/1e6:.1f} MHz  Freq: {args.freq/1e9:.3f} GHz  "
          f"Gain: {args.gain:.0f} dB  Capture: {cap_ms:.0f} ms")
    print(f"  {LINE}")

    # Initialize
    print(f"  [Init] Loading RX chain...")
    rx_npu = init_rx_chain()
    rx_npu.warmup()
    print(f"  [Init] Warmup done")

    # Load TX reference bits if available (script directory)
    tx_bits_file = os.path.join(SCRIPT_DIR, 'tx_bits.npy')
    tx_bits = None
    if os.path.exists(tx_bits_file):
        tx_bits = np.load(tx_bits_file).astype(np.int8)
        print(f"  [Ref]  TX bits: {len(tx_bits)} from {tx_bits_file}")

    # USRP
    print(f"  [USRP] Connecting to {USRP_ADDR}...")
    usrp = uhd.usrp.MultiUSRP(USRP_ADDR)
    usrp.set_rx_rate(args.rate)
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq))
    usrp.set_rx_gain(args.gain)
    usrp.set_rx_antenna("RX2")
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("sc16", "sc16"))
    print(f"  [USRP] {usrp.get_mboard_name()} ready")

    rx_buf = np.zeros(args.capture_len * 2, dtype=np.int16)

    print(f"\n  {'#':>5s}  {'Cap':>6s}  {'NPU':>6s}  "
          f"{'Pwr':>7s}  {'Peak':>5s}  {'Clip':>4s}  "
          f"{'Sync':>4s}  {'CFO':>8s}  {'BER':>10s}")
    print(f"  {LINE}")

    n_recv, ber_hist = 0, []

    try:
        while True:
            # Capture
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            cmd.num_samps = args.capture_len
            cmd.stream_now = True
            streamer.issue_stream_cmd(cmd)

            t0 = time.perf_counter()
            md = uhd.types.RXMetadata()
            n_got = 0
            while n_got < args.capture_len:
                n_rx = streamer.recv(rx_buf[n_got * 2:], md)
                if md.error_code not in (uhd.types.RXMetadataErrorCode.none,
                                          uhd.types.RXMetadataErrorCode.overflow):
                    if args.verbose:
                        print(f"  [Err] {md.error_code}")
                    break
                n_got += n_rx
            dt_cap = (time.perf_counter() - t0) * 1000

            if n_got < args.capture_len // 2:
                print(f"  [Skip] only {n_got} samples")
                continue

            sig = rx_buf[:n_got * 2]

            # Process
            t_start = time.perf_counter()
            try:
                result = rx_process(rx_npu, sig, tx_bits)
            except RuntimeError:
                pwr, pk = signal_stats(sig)
                clip = "CLIP" if pk > 0.95 else " ok "
                print(f"  {n_recv+1:5d}  {dt_cap:5.0f}ms    ERR  "
                      f"{pwr:6.1f}dB  {pk:.3f}  {clip}  FAIL")
                n_recv += 1
                if 0 < args.count <= n_recv:
                    break
                continue
            dt_proc = (time.perf_counter() - t_start) * 1000

            t0 = time.perf_counter()
            pwr, pk = signal_stats(sig)
            clip = "CLIP" if pk > 0.95 else " ok "
            dt_stats = (time.perf_counter() - t0) * 1000

            n_recv += 1
            sync_ok = 'bits' in result
            ber = result.get('ber', -1)
            cfo = result.get('frac_cfo', 0.0)
            if ber >= 0:
                ber_hist.append(ber)

            sync_str = ' OK ' if sync_ok else 'MISS'
            ber_str = f"{ber:.6f}" if ber >= 0 else "     N/A"
            cfo_str = f"{cfo:+7.0f}Hz" if sync_ok else "     N/A"

            # Per-module timing from C++
            t = result.get('timing', {})
            h2d   = t.get('h2d_cvt',0) + t.get('h2d_dma',0)
            sync  = t.get('sync_p1',0) + t.get('sync_p2',0)
            cfo   = t.get('sync_p3',0) + t.get('cfo',0)     # estimation + compensation
            eq    = t.get('eq',0) + t.get('ext',0)           # pilot + LS + data extraction
            total = t.get('total', 0)

            print(f"  {n_recv:5d}  {dt_cap:5.0f}ms  {dt_proc:5.0f}ms  "
                  f"{pwr:6.1f}dB  {pk:.3f}  {clip}  {sync_str}  {cfo_str}  {ber_str}")
            print(f"         H2D:{h2d:5.1f}  Sync:{sync:5.1f}  "
                  f"CFO:{cfo:4.1f}  RRC:{t.get('rrc',0):4.1f}  "
                  f"FFT:{t.get('fft',0):4.1f}  EQ:{eq:4.1f}  "
                  f"QAM:{t.get('qam',0):4.1f}  "
                  f"LDPC:{t.get('ldpc',0):5.1f}  "
                  f"Total:{h2d+sync+total:5.1f}ms")

            if 0 < args.count <= n_recv:
                break

    except KeyboardInterrupt:
        print(f"\n  [Stop] Ctrl+C")

    # Summary
    print(f"\n  {'=' * W}")
    print(f"  Frames: {n_recv}", end="")
    if ber_hist:
        good = sum(1 for b in ber_hist if b < 0.01)
        print(f"  |  Sync: {len(ber_hist)}/{n_recv}"
              f"  |  BER mean: {np.mean(ber_hist):.6f}"
              f"  |  BER<1%: {good}/{len(ber_hist)}")
    else:
        print()
    print(f"  {'=' * W}\n")
    rx_npu.cleanup()


if __name__ == '__main__':
    main()