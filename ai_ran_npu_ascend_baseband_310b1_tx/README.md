# Ascend-RAN TX

**NPU-accelerated OFDM baseband transmitter on Ascend 310B1**

Complete over-the-air OFDM transmitter verified on an edge NPU (Orange Pi AI Pro) with USRP X300 SDR. All baseband DSP runs as 5 custom AscendC operators — no PyTorch, no GPU.

**Key results:** BER=0 over 130+ OTA frames (verified by [RX](../ascend_baseband_310b1_rx)), 38.6ms end-to-end baseband generation.

## Architecture

```
                      info bits (786688)
                           │
                           ▼
┌──────────────────────────────────────────┐
│  Ascend 310B1 NPU (8 AI Cores)           │
│                                          │
│  LDPC (×13 batches) ── QAM64 ── IFFT ×2 │
│                                    │     │
│                      ifft_postproc ◄┘    │
│                           │              │
│                           ▼              │
│                      RRC upsample        │
└──────────────────────────────────────────┘
                           │
                           ▼  ARM NEON 4-phase interleave
                       USRP X300 (sc16, 5MHz)
```

## Performance

Measured over 150+ consecutive frames, Ascend 310B1 + USRP X300.

| Module | Includes | Avg (ms) |
|--------|----------|----------|
| LDPC | Async batch encode ×13 + D2H | 3.1 |
| QAM | Host bit-pack + 64-QAM modulation | 6.8 |
| IFFT | IDFT ×2 + post-processing | 9.1 |
| RRC | Prep (ZC + overlap layout) + upsampling | 11.7 |
| Intlv | D2H + NEON vst4 4-phase interleave | 8.0 |
| **End-to-end** | **bit input → baseband out** | **38.6** |

```
  #50  260ms   3.8 FPS    13.1s
        LDPC:  3.1  QAM:  6.8  IFFT:  9.1  RRC: 11.7  Intlv:  8.0  Total: 38.6ms
```

## System Parameters

| Parameter | Value |
|-----------|-------|
| FFT size | 256 |
| CP length | 16 |
| Oversampling | 4× |
| OFDM symbols | 1192 |
| Data subcarriers | 220 |
| Pilot subcarriers | 16 |
| Modulation | 64-QAM |
| Channel coding | LDPC (512, 256), rate 1/2 |
| Sample rate | 5 MHz |
| Sync sequence | Zadoff-Chu (N=256, u=1) |
| Throughput | 2.97 Mbps |

## Files

```
kernels/                          # 5 AscendC operators
├── ldpc_encode.cpp               #   GF(2) Matmul encoding
├── qam64_modulation.cpp          #   64-QAM mapping (stride=224 output)
├── ofdm_ifft.cpp                 #   IDFT via Matmul (cos/sin)
├── ifft_postproc.cpp             #   td_r=o0-o3, td_i=o1+o2
└── up_sample_rrc.cpp             #   Polyphase RRC (4 phases × R/I)

pybind11.cpp                      # C++ host interface (NEON + ACL)
tx_usrp_npu.py                    # Real-time USRP transmitter
gen_tx_signal.py                  # Offline signal generator (no USRP)
scripts/gen_matrices.py           # Precomputed matrix generator

input/                            # Precomputed matrices
```

## Optimizations

- **LDPC async batching:** 13 kernel launches queued, single final sync (~3ms)
- **Host bit pack:** ARM C++ stride-6 pack is 5× faster than NPU scalar (4.7ms vs 21ms)
- **QAM stride alignment:** QAM writes directly at stride=224, eliminates host pad step (~2.7ms saved)
- **Dual IFFT workspace:** Pre-uploaded pilot bias, zero inter-call sync (~2ms saved)
- **ifft_postproc kernel:** Replaces 4× D2H + host merge (~30ms saved)
- **NEON vst4 interleave:** 4-phase ARM interleave, 6× faster than NPU scalar (8ms vs 53ms)
- **NEON RMS:** `vmlaq_f32` vectorized sum-of-squares for ZC scaling

## Quick Start

### Build

```bash
cd ~/TXSYSTEM
bash run_pybind.sh
```

### Generate matrices

```bash
python3 scripts/gen_matrices.py --output_dir input
```

### Transmit

```bash
# Real-time streaming (stops on Ctrl+C)
python3 tx_usrp_npu.py --gain 0

# Fixed number of frames
python3 tx_usrp_npu.py --gain 0 --count 100

# Generate signal without USRP (for offline testing)
python3 tx_usrp_npu.py --no-usrp
```

The reference TX bits are saved to `tx_bits.npy` for RX BER verification.

### Generate offline signal

```bash
# Clean baseband
python3 gen_tx_signal.py --output-dir ./

# With AWGN
python3 gen_tx_signal.py --snr 20 --output-dir ./

# With CFO
python3 gen_tx_signal.py --cfo 1000 --output-dir ./

# Save in sc16 for direct USRP replay
python3 gen_tx_signal.py --sc16
```

Output files (`.npy`):
| File | Shape | Description |
|------|-------|-------------|
| `tx_bits.npy` | [786688] | Information bits (int8) |
| `tx_baseband.npy` | [~1.3M] | Complex64 baseband waveform |
| `tx_baseband_sc16.npy` | [~2.6M] | Optional sc16 (int16 I/Q) |

### Example output

```
     #     Send     FPS   Elapsed
  ------------------------------------------------------------------------------
      1    249ms    4.0      0.2s
         LDPC:  3.1  QAM:  6.8  IFFT:  9.1  RRC: 11.7  Intlv:  8.0  Total: 38.6ms
     50    260ms    3.8     13.1s
         LDPC:  3.1  QAM:  6.8  IFFT:  9.1  RRC: 11.7  Intlv:  8.0  Total: 38.6ms
    100    261ms    3.8     26.2s
         LDPC:  3.1  QAM:  6.8  IFFT:  9.1  RRC: 11.7  Intlv:  8.0  Total: 38.6ms
```

## Design Notes

### Why some operations stay on ARM host

The chain is ~52% NPU compute / 48% host glue, not because ARM is the target but because data rearrangement (bit packing, 4-phase interleaving) has no efficient vector mapping on the Da Vinci architecture. Empirically:

| Operation | NPU scalar | ARM (cache / NEON) | Ratio |
|-----------|-----------|--------------------|-------|
| LDPC bit-pack (stride 6) | 21 ms | 4.7 ms | 4.5× |
| 4-phase RRC interleave | 53 ms | 8.0 ms | 6.6× |

PipeBarrier overhead (~100μs per row, ×1267 rows × 2 channels ÷ 8 cores) dominates when operations cannot be vectorized. ARM Cortex-A55 with L1/L2 cache handles stride patterns natively.

### Why IFFT uses cos/sin decomposition

Rather than a complex Matmul, the IDFT is factored into two real Matmul ops (one per cos/sin basis), combined by `ifft_postproc`. This doubles the IFFT stage but enables per-call bias injection (for pilot insertion) via Matmul's built-in bias path, saving a separate pilot-insertion kernel.

## Hardware

- **NPU:** Orange Pi AI Pro (Ascend 310B1, 8 AI Cores, 8GB)
- **SDR:** Ettus USRP X300 (10 GbE)
- **Setup:** Two boards — Board A (TX, this repo), Board B ([RX](../ascend_baseband_310b1_rx))

## License

MIT
