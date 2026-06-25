# GNSS Spoofing Detection – Parallelized CAF Acceleration

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A high-performance, parallelized implementation of the **Cross-Ambiguity Function (CAF)** for **real-time GNSS spoofing detection**. This project implements three distinct approaches—Sequential, OpenMP (CPU), and CUDA (GPU)—to demonstrate the power of parallel computing in critical infrastructure security.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [PCAM Design Methodology](#pcam-design-methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Results](#performance-results)
- [Spoofing Detection](#spoofing-detection)
- [Citation](#citation)
- [License](#license)

---

## Overview

Global Navigation Satellite Systems (GNSS) are vulnerable to **spoofing attacks**—where counterfeit signals deceive a receiver into computing false position or time. This project addresses the computational bottleneck of the CAF, which requires **O(D × N²)** operations (13 billion per 1 ms at 25 Msps), by implementing:

| Implementation | Complexity | Hardware |
|:---|:---|:---|
| **Sequential** | O(D × N²) | CPU (1 core) |
| **OpenMP** | O(D × N²) parallelized | CPU (multi-core) |
| **CUDA** | O(N log N) per Doppler bin | GPU (NVIDIA) |

The system achieves up to **49,285× speedup** over the sequential baseline on a Tesla V100 GPU.

---

## Key Features

- **Structure-of-Arrays (SoA)** memory layout for coalesced GPU access
- **Batched FFT** processing all Doppler bins in parallel
- **OpenMP** CPU parallelization with SIMD vectorization
- **CUDA** acceleration with shared memory and pinned memory
- **Synthetic signal test** to validate algorithm correctness
- **Scaling analysis** (5–25 Msps) with real-time compliance plots
- **Thread scaling** and **H2D vs kernel** profiling
- **Spoofing detection** via multi-peak anomaly analysis
- **IEEE-style** results ready for publication

---

## PCAM Design Methodology

The project follows the **PCAM** (Partitioning, Communication, Agglomeration, Mapping) parallel design methodology:

| PCAM Step | OpenMP Implementation | CUDA Implementation |
|:---|:---|:---|
| **Partitioning** | `prange(D)` – one task per Doppler bin | Batched FFT – all Doppler bins in parallel |
| **Communication** | Shared memory (NumPy arrays) | `cp.asarray` / `cp.asnumpy` (H2D/D2H) |
| **Agglomeration** | Chunked static scheduling | Batched FFT (`axis=1`) |
| **Mapping** | CPU cores (12–80 threads) | GPU SMs (`Device(0)`) |

---

## Project Structure
GNSS_Spoofing_Detection_CAF/
├── data/
│ └── USRP_GPS_PRN30.bin # TEXBAT dataset (not included)
├── results/
│ ├── csv/ # Performance data (CSV)
│ ├── plots/ # Generated figures
│ └── logs/ # Benchmark logs
├── src/
│ └── gnss_caf_pcam.py # Main Python implementation
├── scripts/
│ ├── plot_correlation.py # Standalone plotting utilities
│ └── plot_speedup.py
├── README.md
└── requirements.txt


---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/GNSS_Spoofing_Detection_CAF.git
cd GNSS_Spoofing_Detection_CAF

2. Set Up a Virtual Environment
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Verify CUDA
nvidia-smi
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0))"

Note: This project requires CUDA 12.x and a compatible NVIDIA GPU. For CPU-only testing, set FORCE_CPU=1 environment variable.

Usage
Quick Start

python src/gnss_caf_pcam.py

========================================
GNSS Spoofing Detection - PCAM CAF
========================================
Runs per test: 3
GPU: Quadro T1000 (CC 7.5)
Free GPU memory: 3294 MB

Menu Options
MENU:
  1. Sequential
  2. OpenMP
  3. CUDA
  4. ALL (compare all on one size)
  5. Scaling Analysis (5k to 25k, all algorithms)
  6. Thread Scaling Test (OpenMP only)
  7. CUDA H2D vs Kernel Profile

  Citation
  If you use this work, please cite:

  @techreport{Bacha2026GNSS,
    author = {Habtamu Bacha},
    title = {Parallelized Real-Time Cross-Ambiguity Function (CAF) Acceleration for GNSS Spoofing Detection},
    institution = {[Your Institution]},
    year = {2026}}
Contact
Habtamu Bacha – GitHub

Project Link: https://github.com/HabtamuBacha/GNSS_Spoofing_Detection_CAF