#!/usr/bin/env python3
"""
capture_usrp.py — USRP Signal Capture Tool

Captures raw IQ samples from USRP X300 and saves to .npy files
for offline analysis with demo_rx_intermediate.py.

Usage:
  python3 capture_usrp.py                           # single capture
  python3 capture_usrp.py --count 10                 # 10 captures
  python3 capture_usrp.py --count 10 --interval 0.5  # 10 captures, 0.5s apart
  python3 capture_usrp.py --gain 15 --freq 2.45e9    # custom RF settings
  python3 capture_usrp.py --output-dir ./captures    # custom output directory

Output format: complex64 .npy (compatible with demo_rx_intermediate.py)

Part of Ascend-RAN: NPU-accelerated OFDM baseband processing
Target: Ascend 310B1 (Orange Pi AI Pro) + USRP X300
SPDX-License-Identifier: MIT
"""
import uhd
import numpy as np
import os
import sys
import time
import argparse

# Defaults
USRP_ADDR       = "addr=192.168.10.3"
RX_FREQ          = 3e9
RX_RATE          = 5e6
RX_GAIN          = 20
CAPTURE_SAMPLES  = 2800000


def signal_stats(buf):
    """Compute signal statistics from complex64 buffer."""
    pwr_linear = np.mean(np.abs(buf)**2)
    pwr_dbfs = 10 * np.log10(pwr_linear + 1e-20)
    peak = np.max(np.abs(buf))
    return pwr_dbfs, peak


def main():
    parser = argparse.ArgumentParser(description='USRP Signal Capture Tool')
    parser.add_argument('--addr', default=USRP_ADDR, help='USRP address')
    parser.add_argument('--freq', type=float, default=RX_FREQ, help='Center frequency (Hz)')
    parser.add_argument('--rate', type=float, default=RX_RATE, help='Sample rate (Hz)')
    parser.add_argument('--gain', type=float, default=RX_GAIN, help='RX gain (dB)')
    parser.add_argument('--samples', type=int, default=CAPTURE_SAMPLES, help='Samples per capture')
    parser.add_argument('--count', type=int, default=1, help='Number of captures')
    parser.add_argument('--interval', type=float, default=0.0, help='Interval between captures (s)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--prefix', default='usrp_cap', help='Output filename prefix')
    parser.add_argument('--format', choices=['complex64', 'sc16'], default='complex64',
                        help='Save format (complex64 for analysis, sc16 for NPU)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cap_ms = args.samples / args.rate * 1000

    W = 60
    print(f"\n  {'=' * W}")
    print(f"  {'USRP Signal Capture Tool':^{W}}")
    print(f"  {'=' * W}")
    print(f"  USRP:     {args.addr}")
    print(f"  Freq:     {args.freq/1e9:.3f} GHz")
    print(f"  Rate:     {args.rate/1e6:.1f} MHz")
    print(f"  Gain:     {args.gain:.0f} dB")
    print(f"  Samples:  {args.samples} ({cap_ms:.0f} ms)")
    print(f"  Captures: {args.count}")
    print(f"  Format:   {args.format}")
    print(f"  Output:   {args.output_dir}/")
    print(f"  {'-' * W}")

    # Connect to USRP
    print(f"\n  [USRP] Connecting to {args.addr}...")
    usrp = uhd.usrp.MultiUSRP(args.addr)
    usrp.set_rx_rate(args.rate)
    usrp.set_rx_freq(uhd.types.TuneRequest(args.freq))
    usrp.set_rx_gain(args.gain)
    usrp.set_rx_antenna("RX2")

    # Use fc32 for complex64 output, sc16 for raw
    if args.format == 'sc16':
        streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("sc16", "sc16"))
        rx_buf = np.zeros(args.samples * 2, dtype=np.int16)
    else:
        streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
        rx_buf = np.zeros(args.samples, dtype=np.complex64)

    print(f"  [USRP] {usrp.get_mboard_name()} ready")
    print(f"         Actual rate: {usrp.get_rx_rate()/1e6:.3f} MHz")
    print(f"         Actual freq: {usrp.get_rx_freq()/1e9:.6f} GHz")
    print(f"         Actual gain: {usrp.get_rx_gain():.1f} dB")

    # Capture loop
    print(f"\n  {'#':>4s}  {'Time':>8s}  {'Pwr':>8s}  {'Peak':>6s}  {'File'}")
    print(f"  {'-' * W}")

    for cap_idx in range(args.count):
        # Issue stream command
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        cmd.num_samps = args.samples
        cmd.stream_now = True
        streamer.issue_stream_cmd(cmd)

        # Receive
        t0 = time.perf_counter()
        md = uhd.types.RXMetadata()
        n_got = 0

        if args.format == 'sc16':
            while n_got < args.samples:
                n_rx = streamer.recv(rx_buf[n_got * 2:], md)
                if md.error_code not in (uhd.types.RXMetadataErrorCode.none,
                                          uhd.types.RXMetadataErrorCode.overflow):
                    print(f"  [Err] {md.error_code}")
                    break
                n_got += n_rx
        else:
            while n_got < args.samples:
                n_rx = streamer.recv(rx_buf[n_got:], md)
                if md.error_code not in (uhd.types.RXMetadataErrorCode.none,
                                          uhd.types.RXMetadataErrorCode.overflow):
                    print(f"  [Err] {md.error_code}")
                    break
                n_got += n_rx

        dt = (time.perf_counter() - t0) * 1000

        # Stats
        if args.format == 'sc16':
            r = rx_buf[:n_got*2:2].astype(np.float32) / 32768.0
            i = rx_buf[1:n_got*2:2].astype(np.float32) / 32768.0
            pwr = 10 * np.log10(np.mean(r**2 + i**2) + 1e-20)
            peak = np.sqrt(np.max(r**2 + i**2))
        else:
            pwr, peak = signal_stats(rx_buf[:n_got])

        # Filename
        if args.count == 1:
            fname = f"{args.prefix}.npy"
        else:
            fname = f"{args.prefix}_{cap_idx:04d}.npy"
        fpath = os.path.join(args.output_dir, fname)

        # Save
        if args.format == 'sc16':
            np.save(fpath, rx_buf[:n_got * 2])
        else:
            np.save(fpath, rx_buf[:n_got])

        clip = "CLIP" if peak > 0.95 else ""
        print(f"  {cap_idx+1:4d}  {dt:6.0f}ms  {pwr:6.1f}dB  {peak:.3f} {clip:>4s}  {fname}")

        # Interval
        if args.interval > 0 and cap_idx < args.count - 1:
            time.sleep(args.interval)

    # Summary
    fsize = os.path.getsize(fpath)
    print(f"\n  {'-' * W}")
    print(f"  Done. {args.count} capture(s) saved to {args.output_dir}/")
    print(f"  File size: {fsize/1e6:.1f} MB each")
    print(f"  {'=' * W}\n")

    # Usage hint
    if args.count == 1:
        print(f"  Next steps:")
        print(f"    python3 demo_rx_intermediate.py --capture {fpath} --save-plots")
        print(f"    python3 demo_rx_intermediate.py --capture {fpath} --dump-all")
        print()


if __name__ == '__main__':
    main()
