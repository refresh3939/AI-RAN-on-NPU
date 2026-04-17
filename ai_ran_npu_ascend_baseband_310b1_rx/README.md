# Ascend-RAN

**NPU-accelerated OFDM baseband processing on Ascend 310B1**

Complete over-the-air OFDM transceiver verified on an edge NPU (Orange Pi AI Pro) with USRP X300 SDR. All baseband DSP runs as 10 custom AscendC operators — no PyTorch, no GPU.

**Key results:** BER=0 over 130+ OTA frames, 46ms NPU-only latency, 63ms end-to-end.

## Architecture

```
USRP X300 (sc16, 5MHz)
    │
    ▼  ARM NEON sc16→f16
┌──────────────────────────────────────────┐
│  Ascend 310B1 NPU (8 AI Cores)           │
│                                          │
│  Sync ─── CFO ─── RRC ─── FFT           │
│                              │           │
│              EQ ◄────────────┘           │
│              │                           │
│           QAM ─── LDPC (×13 batches)     │
│              │                           │
│              ▼                           │
│         decoded bits                     │
└──────────────────────────────────────────┘
    │
    ▼  BER = 0.000000
```

## Performance

Measured over 130+ consecutive OTA frames (BER=0), Ascend 310B1 + USRP X300.

| Module | Includes | Avg (ms) |
|--------|----------|----------|
| H2D | NEON sc16→f16 + DMA upload | 16.3 |
| Sync | Energy scan + cross-correlation | 10.9 |
| CFO | Estimation (CP) + compensation | 2.8 |
| RRC | Matched filter + downsample | 2.3 |
| FFT | DFT ×2 + post-processing | 4.1 |
| EQ | Pilot ext + LS-ZF + data ext | 4.7 |
| QAM | 64-QAM demodulation | 8.3 |
| LDPC | Repack + decode ×13 + D2H | 13.1 |
| **NPU-only** | **Sync → LDPC** | **46.2** |
| **End-to-end** | **H2D → LDPC** | **62.5** |

```
  #128  560ms   68ms  -48.3dB  0.018   ok   OK   -2026Hz  0.000000
        H2D: 15.9  Sync: 10.3  CFO: 2.8  RRC: 2.3  FFT: 4.2  EQ: 4.7  QAM: 8.1  LDPC: 13.1  Total: 59.9ms
```

## System Parameters

| Parameter | Value |
|-----------|-------|
| FFT size | 256 |
| CP length | 16 |
| Oversampling | 4× |
| OFDM symbols | 1192 |
| Data subcarriers | 220 |
| Modulation | 64-QAM |
| Channel coding | LDPC (512, 256) |
| Sample rate | 5 MHz |
| Throughput | 2.97 Mbps |

## Files

```
kernels/                          # 10 AscendC operators (2057 lines)
├── fine_sync.cpp                 #   3-phase sync (8-core parallel)
├── cfo_compensate.cpp            #   CFO compensation (vector cos/sin)
├── rrc_downsample.cpp            #   RRC filter (Matmul Toeplitz)
├── ofdm_fft.cpp                  #   OFDM DFT (Matmul)
├── fft_postproc.cpp              #   FFT post-processing
├── extract_subcarriers.cpp       #   Pilot subcarrier extraction
├── ls_equalizer.cpp              #   LS estimation + ZF equalization
├── data_extract_mm.cpp           #   Data extraction (Matmul permutation)
├── qam64_demod.cpp               #   64-QAM demodulation
└── ldpc_decode.cpp               #   LDPC bit-flipping decoder

pybind11.cpp                      # C++ host interface (NEON + ACL)
rx_usrp_npu.py                    # Real-time USRP receiver
capture_usrp.py                   # Signal capture tool
demo_rx_intermediate.py           # Intermediate output extraction

input/                            # Precomputed matrices (gen_matrices.py)
```

## Quick Start

### Build

```bash
cd ~/aclsystem
bash run_pybind.sh
```

### Generate matrices

```bash
python3 gen_matrices.py --output_dir input
```

### Run

```bash
# Real-time reception
python3 rx_usrp_npu.py --gain 15 --count 100
```

### Capture signals

```bash
# Single capture
python3 capture_usrp.py --gain 15

# Batch capture (10 frames, 0.5s interval)
python3 capture_usrp.py --gain 15 --count 10 --interval 0.5 --output-dir captures/

# Custom RF settings
python3 capture_usrp.py --freq 2.45e9 --rate 5e6 --gain 20 --samples 2800000
```

Output: `.npy` files (complex64), one per capture.

### Offline analysis

```bash
# Full analysis with plots
python3 demo_rx_intermediate.py --capture captures/usrp_cap_0003.npy --save-plots

# Export all intermediate data for MATLAB/Python analysis
python3 demo_rx_intermediate.py --capture captures/usrp_cap_0003.npy --dump-all

# Both
python3 demo_rx_intermediate.py --capture captures/usrp_cap_0003.npy --save-plots --dump-all
```

Generated plots (`plots/` directory):
| File | Content |
|------|---------|
| `1_time_domain.png` | Received waveform + sync position |
| `2_constellation.png` | 64-QAM before/after equalization |
| `3_spectrum.png` | Subcarrier power spectrum |
| `4_evm.png` | EVM per OFDM symbol |

Exported data (`dump/` directory):
| File | Shape | Description |
|------|-------|-------------|
| `comp.npy` | [1297920] | After CFO compensation |
| `rrc.npy` | [1192, 256] | After RRC filter |
| `freq.npy` | [1192, 256] | Frequency domain |
| `eq.npy` | [1192, 256] | After equalization |
| `qam.npy` | [1192, 220] | QAM constellation points |
| `pilot.npy` | [1192, 16] | Pilot observations |

### Example output

```
     92  560ms   68ms  -48.3dB  0.017   ok   OK   -2033Hz  0.000000
         H2D: 16.2  Sync: 10.7  CFO: 2.9  RRC: 2.3  FFT: 4.1  EQ: 4.7  QAM: 7.0  LDPC: 13.6  Total: 61.5ms
```

## Hardware

- **NPU:** Orange Pi AI Pro (Ascend 310B1, 8 AI Cores, 8GB)
- **SDR:** Ettus USRP X300 (10 GbE)
- **Setup:** Two boards — Board A (TX/compile), Board B (RX)

## License

MIT
