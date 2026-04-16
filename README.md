# Ascend-RAN

**NPU-accelerated OFDM transceiver on Ascend 310B1**

First over-the-air verification of a complete OFDM ISAC transceiver on an edge NPU (Orange Pi AI Pro, Ascend 310B1) paired with a USRP X300 SDR. All baseband DSP — from LDPC encode to OTA transmission, and from sample capture to decoded bits — runs as 15 custom AscendC operators across two boards. No PyTorch, no GPU.

**Key results:**
- BER = 0 over 130+ OTA frames (real 2.45 GHz RF link)
- TX: 38.6 ms end-to-end baseband generation
- RX: 62.5 ms end-to-end receive & decode (46.2 ms NPU-only)
- Throughput: 2.97 Mbps @ 5 MHz sample rate

## Overview

```
┌──────────────────────┐         ┌──────────────────────┐
│  Board A (TX)        │         │  Board B (RX)        │
│  Orange Pi AI Pro    │         │  Orange Pi AI Pro    │
│  Ascend 310B1 NPU    │         │  Ascend 310B1 NPU    │
│                      │         │                      │
│   info bits          │         │    sc16 samples      │
│       │              │         │        │             │
│       ▼              │         │        ▼             │
│   LDPC encode        │         │    Sync + CFO        │
│   QAM modulation     │         │    RRC + FFT         │
│   OFDM IFFT          │         │    EQ + QAM demod    │
│   RRC upsample       │         │    LDPC decode       │
│       │              │         │        │             │
│       ▼              │         │        ▼             │
└───────┼──────────────┘         └────────┼─────────────┘
        │                                 ▲
        │         2.45 GHz RF             │
        ▼                                 │
   USRP X300  ──────────────────────► USRP X300
   (TX antenna)                      (RX antenna)
```

## Repository Layout

```
ascend_baseband_310b1/
├── README.md                          ← you are here
├── LICENSE                            ← MIT
│
├── ascend_baseband_310b1_tx/          ← Transmitter chain
│   ├── README.md                      ←   TX documentation
│   ├── kernels/                       ←   5 AscendC operators
│   │   ├── ldpc_encode.cpp
│   │   ├── qam64_modulation.cpp
│   │   ├── ofdm_ifft.cpp
│   │   ├── ifft_postproc.cpp
│   │   └── up_sample_rrc.cpp
│   ├── pybind11.cpp
│   ├── tx_usrp_npu.py                 ←   real-time USRP transmitter
│   └── gen_tx_signal.py               ←   offline signal generator
│
└── ascend_baseband_310b1_rx/          ← Receiver chain
    ├── README.md                      ←   RX documentation
    ├── kernels/                       ←   10 AscendC operators
    │   ├── fine_sync.cpp
    │   ├── cfo_compensate.cpp
    │   ├── rrc_downsample.cpp
    │   ├── ofdm_fft.cpp
    │   ├── fft_postproc.cpp
    │   ├── extract_subcarriers.cpp
    │   ├── ls_equalizer.cpp
    │   ├── data_extract_mm.cpp
    │   ├── qam64_demod.cpp
    │   └── ldpc_decode.cpp
    ├── pybind11.cpp
    ├── rx_usrp_npu.py                 ←   real-time USRP receiver
    ├── capture_usrp.py
    └── demo_rx_intermediate.py
```

## End-to-End Performance

Measured over 150+ consecutive OTA frames, 2.45 GHz carrier, 5 MHz sample rate.

### TX chain (38.6 ms)

| Module | Avg (ms) |
|--------|---------:|
| LDPC | 3.1 |
| QAM | 6.8 |
| IFFT | 9.1 |
| RRC | 11.7 |
| Intlv | 8.0 |
| **Total** | **38.6** |

### RX chain (62.5 ms end-to-end, 46.2 ms NPU-only)

| Module | Avg (ms) |
|--------|---------:|
| H2D | 16.3 |
| Sync | 10.9 |
| CFO | 2.8 |
| RRC | 2.3 |
| FFT | 4.1 |
| EQ | 4.7 |
| QAM | 8.3 |
| LDPC | 13.1 |
| **Total** | **62.5** |

## System Parameters

| Parameter | Value |
|-----------|-------|
| FFT size | 256 |
| CP length | 16 |
| Oversampling | 4× |
| OFDM symbols per frame | 1192 |
| Data subcarriers | 220 |
| Pilot subcarriers | 16 |
| Modulation | 64-QAM |
| Channel coding | LDPC (512, 256), rate 1/2 |
| Sample rate | 5 MHz |
| Occupied bandwidth | 1.69 MHz |
| Sync sequence | Zadoff-Chu (N=256, u=1) |
| Throughput | 2.97 Mbps |

## Quick Start

### Hardware setup

- **Board A (TX):** Orange Pi AI Pro (Ascend 310B1) + USRP X300
- **Board B (RX):** Orange Pi AI Pro (Ascend 310B1) + USRP X300
- **Link:** 2.45 GHz RF (configurable), antennas with line-of-sight

### Build & run

```bash
# On Board A (TX)
cd ascend_baseband_310b1_tx
python3 scripts/gen_matrices.py --output_dir input   # generate precomputed matrices
bash run_pybind.sh                                    # compile AscendC kernels
python3 tx_usrp_npu.py --gain 0                       # transmit

# On Board B (RX)
cd ascend_baseband_310b1_rx
python3 gen_matrices.py --output_dir input
bash run_pybind.sh
python3 rx_usrp_npu.py --gain 15 --count 100
```

See [`ascend_baseband_310b1_tx/README.md`](ascend_baseband_310b1_tx/README.md) and [`ascend_baseband_310b1_rx/README.md`](ascend_baseband_310b1_rx/README.md) for detailed build instructions, optimization notes, and per-module explanations.

## Design Philosophy

This repository demonstrates **computational isomorphism** between NPU primitives (Matmul, vector element-wise) and classical baseband DSP operations:

| DSP operation | NPU primitive |
|---------------|---------------|
| LDPC encode / decode | Matmul (GF(2)) |
| QAM modulation / demod | Vector Cast + Muls/Adds |
| OFDM IFFT / FFT | Matmul (DFT matrix) |
| RRC polyphase filter | Matmul (Toeplitz) |
| LS channel estimation | Matmul (pseudo-inverse) |
| Data subcarrier extraction | Matmul (permutation matrix) |
| CFO compensation | Vector cos/sin (angle-sum) |
| Timing sync | Vector correlation |

NPU hardware natively covers all baseband operations — positioning NPU as an unexplored alternative to GPU-based AI-RAN.

## Design Notes

### Boundary between NPU and ARM host

The chain is ~50% NPU compute / 50% ARM host glue. This is not because ARM is the target but because **data rearrangement** (bit packing, phase interleaving, stride-N memory patterns) has no efficient vector mapping on the Da Vinci architecture. Empirically:

| Operation | NPU scalar | ARM (cache / NEON) | Ratio |
|-----------|-----------|--------------------|-------|
| LDPC bit-pack (stride 6) | 21 ms | 4.7 ms | 4.5× |
| 4-phase RRC interleave | 53 ms | 8.0 ms | 6.6× |
| sc16 → f16 conversion | N/A | 2.1 ms (NEON vcvt_f16_f32) | — |

PipeBarrier overhead per row dominates when operations cannot be vectorized. ARM Cortex-A55 with L1/L2 cache handles stride patterns natively, so we keep these on the host.

### Why IFFT/FFT uses cos/sin decomposition

Rather than a complex Matmul, the DFT is factored into two real Matmul ops (one per cos/sin basis), combined by a post-processing kernel. This doubles the Matmul stage but:
- Enables per-call bias injection (pilot insertion on TX, delay compensation on RX) via Matmul's built-in bias path
- Saves a separate pilot-insertion kernel
- Simplifies numeric verification against reference implementations

## Hardware

- **NPU:** Orange Pi AI Pro (Ascend 310B1, 8 AI Cores, 8 GB)
- **SDR:** Ettus USRP X300 (10 GbE connection)
- **CPU:** ARM Cortex-A55 (on SoC)

## Citation

If this work is useful for your research, please cite:

```
@article{ascend-ran-2026,
  title={Ascend-RAN: NPU-Accelerated OFDM Baseband Processing on Edge Devices},
  author={...},
  journal={IEEE Communications Magazine (submitted)},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
