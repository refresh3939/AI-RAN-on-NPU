# AI-RAN on NPU: OFDM Baseband on Ascend 310B1

**NPU-accelerated OFDM transceiver on Ascend 310B1**

First over-the-air demonstration of a complete OFDM baseband transceiver running entirely on an edge NPU (Orange Pi AI Pro, Ascend 310B1) paired with USRP X300 SDRs. All baseband DSP — from LDPC encode to OTA transmission, and from sample capture to decoded bits — runs as 15 custom AscendC operators across two boards. No PyTorch, no GPU.

**Key results:**
- BER = 0 over 130+ consecutive OTA frames (real 2.45 GHz RF link)
- NPU-only latency: 30.7 ms (TX) / 46.2 ms (RX)
- End-to-end latency: 38.6 ms (TX) / 62.5 ms (RX), including ARM host operations
- Throughput: 2.97 Mbps @ 5 MHz sample rate, 1.69 MHz occupied bandwidth

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
NPU-baseband-OFDM/
├── README.md                                 ← you are here
├── LICENSE                                   ← MIT
│
├── ai_ran_npu_ascend_baseband_310b1_tx/      ← Transmitter chain
│   ├── kernels/                              ←   5 AscendC operators
│   │   ├── ldpc_encode.cpp
│   │   ├── qam64_modulation.cpp
│   │   ├── ofdm_ifft.cpp
│   │   ├── ifft_postproc.cpp
│   │   └── up_sample_rrc.cpp
│   ├── scripts/
│   ├── tiling/
│   ├── pybind11.cpp
│   ├── tx_usrp_npu.py                        ←   real-time USRP transmitter
│   └── gen_tx_signal.py                      ←   offline signal generator
│
└── ai_ran_npu_ascend_baseband_310b1_rx/      ← Receiver chain
    ├── kernels/                              ←   10 AscendC operators
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
    ├── cmake/
    ├── scripts/
    ├── tiling/
    ├── pybind11.cpp
    ├── rx_usrp_npu.py                        ←   real-time USRP receiver
    ├── capture_usrp.py
    └── demo_rx_intermediate.py
```

## End-to-End Performance

Measured over 130+ consecutive OTA frames at 2.45 GHz carrier, 5 MHz sample rate.

### TX chain

| Module | NPU (ms) |
|--------|---------:|
| LDPC encode  | 3.1  |
| QAM modulation | 6.8 |
| OFDM IFFT    | 9.1  |
| RRC upsample | 11.7 |
| **NPU total**      | **30.7** |
| ARM host (bit-packing, interleaving) | 7.9 |
| **End-to-end** | **38.6** |

### RX chain

| Module | NPU (ms) |
|--------|---------:|
| Sync         | 10.9 |
| CFO compensation | 2.8 |
| RRC downsample | 2.3 |
| OFDM FFT     | 4.1  |
| LS+ZF equalization | 4.7 |
| QAM demod    | 8.3  |
| LDPC decode  | 13.1 |
| **NPU total**      | **46.2** |
| Host-to-device (H2D) transfer | 16.3 |
| **End-to-end** | **62.5** |

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
cd ai_ran_npu_ascend_baseband_310b1_tx
python3 scripts/gen_matrices.py --output_dir input
bash run_pybind.sh
python3 tx_usrp_npu.py --gain 0

# On Board B (RX)
cd ai_ran_npu_ascend_baseband_310b1_rx
python3 scripts/gen_matrices.py --output_dir input
bash run_pybind.sh
python3 rx_usrp_npu.py --gain 15 --count 100
```

Note: USRP IP addresses are currently hardcoded in `tx_usrp_npu.py` (`192.168.20.2`) and `rx_usrp_npu.py` (`192.168.10.3`). Modify these to match your USRP setup.

## Design Philosophy

This repository demonstrates **computational isomorphism** between NPU primitives (matrix multiplication, vector element-wise) and classical baseband DSP operations:

| DSP operation | NPU primitive |
|---------------|---------------|
| LDPC encode / decode | Matmul (GF(2)) |
| QAM modulation / demod | Vector cast + muls/adds |
| OFDM IFFT / FFT | Matmul (DFT matrix) |
| RRC polyphase filter | Matmul (Toeplitz) |
| LS channel estimation | Matmul (pseudo-inverse) |
| Data subcarrier extraction | Matmul (permutation matrix) |
| CFO compensation | Vector cos/sin (angle-sum) |
| Timing sync | Vector correlation |

NPU hardware natively covers all baseband operations — positioning the NPU as an unexplored alternative to GPU-based AI-RAN.

## Design Notes

### Boundary between NPU and ARM host

The chain is ~50% NPU compute / 50% ARM host glue. This is not because ARM is the target, but because **data rearrangement** (bit packing, phase interleaving, stride-N memory patterns) has no efficient vector mapping on the Da Vinci architecture. Empirically:

| Operation | NPU scalar | ARM (cache / NEON) | Ratio |
|-----------|-----------|--------------------|-------|
| LDPC bit-pack (stride 6) | 21 ms | 4.7 ms | 4.5× |
| 4-phase RRC interleave | 53 ms | 8.0 ms | 6.6× |
| sc16 → f16 conversion | N/A | 2.1 ms (NEON vcvt_f16_f32) | — |

PipeBarrier overhead per row dominates when operations cannot be vectorized. ARM Cortex-A55 with L1/L2 cache handles stride patterns natively, so we keep these on the host.

### Why IFFT/FFT uses cos/sin decomposition

Rather than a complex Matmul, the DFT is factored into two real Matmul operations (one per cos/sin basis), combined by a post-processing kernel. This doubles the Matmul stage but:
- Enables per-call bias injection (pilot insertion on TX, delay compensation on RX) via Matmul's built-in bias path
- Saves a separate pilot-insertion kernel
- Simplifies numeric verification against reference implementations

## Hardware

- **NPU:** Orange Pi AI Pro (Ascend 310B1, 8 TOPS INT8, 8 W TDP)
- **SDR:** Ettus USRP X300 (10 GbE connection)
- **CPU:** ARM Cortex-A55 (on SoC)

## License

MIT — see [LICENSE](LICENSE).

## Contact

Shilong Zhang, Nanjing University
Email: refresh3939@gmail.com