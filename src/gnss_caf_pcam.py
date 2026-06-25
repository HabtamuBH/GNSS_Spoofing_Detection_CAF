#!/usr/bin/env python3
"""
GNSS Spoofing Detection – PCAM Parallelized CAF
Academic Implementation

This code implements a parallelized Cross-Ambiguity Function (CAF) for
GNSS spoofing detection using Structure-of-Arrays (SoA) memory architecture,
OpenMP CPU parallelization, and CUDA GPU acceleration.

Author: Habtamu Bacha
Course: Parallelized Real-Time CAF Acceleration for GNSS Spoofing Detection
"""

import os
import sys
import time
import math
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
import psutil

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

NUM_RUNS = 3
DOPPLER_START = -5000.0
DOPPLER_END = 5000.0
DOPPLER_STEP = 500.0
DOPPLER_BINS = np.arange(DOPPLER_START, DOPPLER_END + DOPPLER_STEP, DOPPLER_STEP)
REAL_DATA_FILE = "data/USRP_GPS_PRN30.bin"

# =============================================================================
# CUDA SETUP
# =============================================================================

HAVE_CUDA = False
device_name = "None"
device_cc = "N/A"
free_gpu_mem = 0

try:
    import cupy as cp
    from cupyx.scipy.fft import fft, ifft

    device = cp.cuda.Device(0)
    device_id = device.id
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    device_cc = f"{props['major']}.{props['minor']}"
    free_gpu_mem = cp.cuda.runtime.memGetInfo()[0]
    HAVE_CUDA = True

except Exception as e:
    HAVE_CUDA = False
    raise RuntimeError(f"CUDA is required but not available: {e}")

if os.environ.get('FORCE_CPU', '0') == '1':
    HAVE_CUDA = False
    raise RuntimeError("FORCE_CPU=1 set, but CUDA is required.")

# =============================================================================
# DATA STRUCTURE: STRUCTURE-OF-ARRAYS (SoA)
# =============================================================================

@dataclass
class IQDataSoA:
    """Structure-of-Arrays (SoA) layout for I/Q samples.
    
    Real and Imaginary samples are stored in separate contiguous memory blocks.
    This layout enables memory coalescing on GPUs and improves cache locality on CPUs.
    """
    real: np.ndarray
    imag: np.ndarray

    def __len__(self):
        return len(self.real)


def load_soa_data(filename, max_samples=None):
    """
    Load raw interleaved I/Q samples from a TEXBAT .bin file.
    
    The file is expected to contain interleaved int8 samples: I1, Q1, I2, Q2, ...
    Samples are de-interleaved into separate Real and Imag arrays and normalized to [-1, 1].
    
    Args:
        filename: Path to the .bin file
        max_samples: Maximum number of samples to load (None = load all)
    
    Returns:
        IQDataSoA: Structure-of-Arrays containing real and imag samples
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset not found: {filename}")

    raw = np.fromfile(filename, dtype=np.int8)
    real = raw[0::2].astype(np.float32) / 128.0
    imag = raw[1::2].astype(np.float32) / 128.0

    if max_samples is not None:
        real = real[:max_samples]
        imag = imag[:max_samples]

    return IQDataSoA(real=real, imag=imag)


def generate_prn(prn_id, num_samples):
    """
    Generate GPS C/A code for a given PRN number, upsampled to match the signal length.
    
    The GPS C/A code is generated using two 10-bit LFSRs (G1 and G2) as specified in
    the GPS ICD-GPS-200 standard. The 1023-chip code is upsampled to num_samples.
    
    Args:
        prn_id: PRN number (1-32)
        num_samples: Number of samples to generate
    
    Returns:
        tuple: (replica_real, replica_imag) as numpy arrays
    """
    g2_delays = {
        1: (2,6), 2: (3,7), 3: (4,8), 4: (5,9), 5: (1,9), 6: (2,10),
        7: (1,8), 8: (2,9), 9: (3,10), 10: (2,3), 11: (3,9), 12: (5,6),
        13: (6,9), 14: (4,10), 15: (5,7), 16: (4,9), 17: (5,10), 18: (6,8),
        19: (7,9), 20: (8,10), 21: (1,7), 22: (1,10), 23: (2,4), 24: (3,4),
        25: (5,8), 26: (6,10), 27: (7,8), 28: (8,9), 29: (3,5), 30: (4,7),
        31: (1,4), 32: (2,5)
    }

    if prn_id not in g2_delays:
        raise ValueError(f"PRN {prn_id} not supported.")

    tap1, tap2 = g2_delays[prn_id]
    tap1 -= 1
    tap2 -= 1

    g1_reg = 0x3FF
    g2_reg = 0x3FF
    ca_code = np.zeros(1023, dtype=np.float32)

    for chip in range(1023):
        g1_out = (g1_reg >> 9) & 1
        g2_out = ((g2_reg >> tap1) & 1) ^ ((g2_reg >> tap2) & 1)
        bit = g1_out ^ g2_out
        ca_code[chip] = 1.0 if bit == 0 else -1.0

        # Advance G1 shift register (polynomial x^10 + x^3 + 1)
        g1_feedback = ((g1_reg >> 9) & 1) ^ ((g1_reg >> 2) & 1)
        g1_reg = ((g1_reg << 1) & 0x3FF) | g1_feedback

        # Advance G2 shift register (polynomial x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1)
        g2_feedback = (((g2_reg >> 9) & 1) ^ ((g2_reg >> 8) & 1) ^
                       ((g2_reg >> 7) & 1) ^ ((g2_reg >> 5) & 1) ^
                       ((g2_reg >> 2) & 1) ^ ((g2_reg >> 1) & 1))
        g2_reg = ((g2_reg << 1) & 0x3FF) | g2_feedback

    samples_per_chip = num_samples / 1023.0
    idx = np.floor(np.arange(num_samples) / samples_per_chip).astype(np.int32)
    idx = np.clip(idx, 0, 1022)

    replica_real = ca_code[idx]
    replica_imag = np.zeros_like(replica_real)

    return replica_real, replica_imag

# =============================================================================
# CAF IMPLEMENTATIONS
# =============================================================================

def compute_caf_sequential(rx_real, rx_imag, rep_real, rep_imag,
                           doppler_bins, sample_rate):
    """
    Sequential CAF implementation using triple nested loops.
    
    Complexity: O(D * N^2) where D = number of Doppler bins, N = number of samples.
    
    Args:
        rx_real, rx_imag: Received signal (I/Q)
        rep_real, rep_imag: Local replica
        doppler_bins: Array of Doppler frequencies to search
        sample_rate: Sampling rate in Hz
    
    Returns:
        corr: 2D correlation surface (D x N)
    """
    N = len(rx_real)
    D = len(doppler_bins)
    dt = 1.0 / sample_rate
    corr = np.zeros((D, N), dtype=np.float32)
    t = np.arange(N, dtype=np.float32)

    for d, doppler in enumerate(doppler_bins):
        angle = -2.0 * np.pi * doppler * t * dt
        exp_real = np.cos(angle)
        exp_imag = np.sin(angle)

        for tau in range(N):
            shifted_idx = (t.astype(np.int64) - tau) % N
            r_real = rep_real[shifted_idx]
            r_imag = rep_imag[shifted_idx]

            # Complex multiplication: s(t) * r*(t-tau)
            temp_real = rx_real * r_real - rx_imag * r_imag
            temp_imag = rx_real * r_imag + rx_imag * r_real

            # Multiply by exp(-j*2*pi*doppler*t*dt)
            res_real = temp_real * exp_real - temp_imag * exp_imag
            res_imag = temp_real * exp_imag + temp_imag * exp_real

            sum_real = np.sum(res_real)
            sum_imag = np.sum(res_imag)
            corr[d, tau] = sum_real * sum_real + sum_imag * sum_imag

    return corr


@njit(parallel=True, fastmath=True, cache=True)
def compute_caf_openmp_numba(rx_real, rx_imag, rep_real, rep_imag,
                             doppler_bins, sample_rate):
    """
    OpenMP-parallelized CAF using Numba.
    
    The outer loops (Doppler bins and code phases) are distributed across CPU cores
    using prange. The inner time loop is SIMD-vectorized.
    
    Complexity: O(D * N^2) parallelized across CPU cores.
    """
    N = len(rx_real)
    D = len(doppler_bins)
    dt = 1.0 / sample_rate
    corr = np.zeros((D, N), dtype=np.float32)

    # Precompute circular shift offsets
    base = np.zeros(N, dtype=np.int64)
    for tau in range(N):
        base[tau] = (N - tau) % N

    for d in prange(D):
        doppler = doppler_bins[d]

        # Precompute exponentials for this Doppler bin
        exp_real = np.empty(N, dtype=np.float32)
        exp_imag = np.empty(N, dtype=np.float32)
        for t in range(N):
            angle = -2.0 * np.pi * doppler * t * dt
            exp_real[t] = np.cos(angle)
            exp_imag[t] = np.sin(angle)

        for tau in range(N):
            sum_real = 0.0
            sum_imag = 0.0
            b = base[tau]

            for t in range(N):
                idx = t + b
                if idx >= N:
                    idx -= N

                s_r = rx_real[t]
                s_i = rx_imag[t]
                r_r = rep_real[idx]

                # Complex multiplication: s(t) * r*(t-tau) with r_imag = 0
                temp_r = s_r * r_r
                temp_i = s_i * r_r

                # Multiply by exp(-j*2*pi*doppler*t*dt)
                er = exp_real[t]
                ei = exp_imag[t]
                res_r = temp_r * er - temp_i * ei
                res_i = temp_r * ei + temp_i * er

                sum_real += res_r
                sum_imag += res_i

            corr[d, tau] = sum_real * sum_real + sum_imag * sum_imag

    return corr


def compute_caf_openmp(rx_real, rx_imag, rep_real, rep_imag,
                       doppler_bins, sample_rate):
    """Wrapper for OpenMP-parallelized CAF."""
    return compute_caf_openmp_numba(rx_real, rx_imag, rep_real, rep_imag,
                                    doppler_bins, sample_rate)


def compute_caf_cuda(rx_real, rx_imag, rep_real, rep_imag,
                     doppler_bins, sample_rate):
    """
    CUDA-accelerated CAF using batched FFTs.
    
    The implementation uses CuPy's FFT module to compute the CAF in the frequency domain.
    All Doppler bins are processed in parallel using batched FFTs.
    
    Complexity: O(N log N) per Doppler bin, processed in parallel on GPU.
    """
    if not HAVE_CUDA:
        raise RuntimeError("CUDA is required but not available.")

    N = len(rx_real)
    D = len(doppler_bins)
    dt = 1.0 / sample_rate

    # Estimate GPU memory requirement
    bytes_per_complex = 8
    estimated_bytes = (2 + 5 * D) * N * bytes_per_complex * 2.5

    if estimated_bytes > free_gpu_mem:
        raise MemoryError(
            f"Requested {N} samples would need approximately "
            f"{estimated_bytes // (1024**2)} MB GPU memory, "
            f"but only {free_gpu_mem // (1024**2)} MB is free."
        )

    try:
        # Transfer data to GPU
        rx = cp.asarray(rx_real + 1j * rx_imag, dtype=cp.complex64)
        rep = cp.asarray(rep_real + 1j * rep_imag, dtype=cp.complex64)

        # Forward FFT on received signal (once)
        rx_fft = fft(rx)

        # Precompute phase ramps for all Doppler bins
        t = cp.arange(N, dtype=cp.float32)
        dop = cp.asarray(doppler_bins, dtype=cp.float32)
        angle = 2.0 * cp.pi * cp.outer(dop, t) * dt
        phase = cp.exp(1j * angle)

        # Batched replica with phase ramps
        rep_batch = cp.tile(rep, (D, 1))
        rep_ramped = rep_batch * phase

        # Batched forward FFT on replica (all Doppler bins in parallel)
        rep_fft = fft(rep_ramped, axis=1)

        # Element-wise multiplication: S(f) * conj(R_d(f))
        result_fft = rx_fft * cp.conj(rep_fft)

        # Batched inverse FFT
        corr_complex = ifft(result_fft, axis=1)

        # Compute magnitude squared
        corr = cp.abs(corr_complex) ** 2

        # Synchronize to ensure all GPU operations are complete
        cp.cuda.Stream.null.synchronize()

        # Transfer results back to CPU
        corr = cp.asnumpy(corr).astype(np.float32)

        return corr

    except cp.cuda.memory.OutOfMemoryError as e:
        raise MemoryError(f"CUDA out of memory: {e}. Reduce sample size.")
    except Exception as e:
        raise RuntimeError(f"CUDA computation failed: {e}")

# =============================================================================
# SPOOFING DETECTION
# =============================================================================

def detect_spoofing(corr_surface, doppler_bins, threshold_ratio=0.5, min_sep=10):
    """
    Detect spoofing by analyzing the correlation surface for multi-peak anomalies.
    
    Under normal operation, a single sharp peak denotes authentic signal tracking.
    Under spoofing conditions, a multi-peak anomaly appears.
    
    Args:
        corr_surface: 2D correlation surface (Doppler x Code Phase)
        doppler_bins: Array of Doppler frequencies
        threshold_ratio: Secondary peak must exceed this fraction of main peak to flag spoofing
        min_sep: Minimum code phase separation between peaks (avoids side lobes)
    
    Returns:
        dict: Detection result with status, peak ratio, and peak information
    """
    D, N = corr_surface.shape

    # Find main peak
    main_idx = np.argmax(corr_surface)
    main_d = main_idx // N
    main_tau = main_idx % N
    main_val = corr_surface[main_d, main_tau]

    # Create mask to exclude main peak and its vicinity
    mask = np.ones_like(corr_surface, dtype=bool)
    mask[main_d, main_tau] = False

    tau_min = max(0, main_tau - min_sep)
    tau_max = min(N, main_tau + min_sep + 1)
    mask[main_d, tau_min:tau_max] = False

    # Find secondary peak
    if np.any(mask):
        sec_idx = np.argmax(corr_surface * mask)
        sec_d = sec_idx // N
        sec_tau = sec_idx % N
        sec_val = corr_surface[sec_d, sec_tau]
    else:
        sec_val = 0
        sec_d = -1
        sec_tau = -1

    peak_ratio = sec_val / main_val if main_val > 0 else 0
    is_spoofed = peak_ratio >= threshold_ratio

    result = {
        'is_spoofed': is_spoofed,
        'peak_ratio': peak_ratio,
        'main_peak': {
            'value': main_val,
            'doppler_idx': main_d,
            'code_idx': main_tau,
            'doppler_hz': doppler_bins[main_d] if main_d < len(doppler_bins) else 0
        },
        'secondary_peak': {
            'value': sec_val,
            'doppler_idx': sec_d,
            'code_idx': sec_tau,
            'doppler_hz': doppler_bins[sec_d] if sec_d >= 0 and sec_d < len(doppler_bins) else 0
        },
        'reason': f"Secondary peak is {peak_ratio * 100:.2f}% of main peak"
    }

    return result

# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def time_function(func, *args, runs=NUM_RUNS):
    """
    Execute a function multiple times and return the average execution time.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        runs: Number of executions
    
    Returns:
        tuple: (average_time_ms, last_result)
    """
    times = []
    result = None

    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = sum(times) / runs
    return avg_time, result


def run_algorithm(name, func, rx_r, rx_i, rep_r, rep_i,
                  doppler_bins, num_samples, base_mem):
    """
    Run a single algorithm and return performance metrics.
    
    Args:
        name: Algorithm name (for display)
        func: Algorithm function
        rx_r, rx_i: Received signal
        rep_r, rep_i: Local replica
        doppler_bins: Doppler frequencies
        num_samples: Number of samples
        base_mem: Base memory usage before execution
    
    Returns:
        tuple: (time_ms, rate, gflops, memory_peak, correlation_surface)
    """
    print(f"  Running {name}...", end='', flush=True)

    try:
        t, corr = time_function(func, rx_r, rx_i, rep_r, rep_i,
                                doppler_bins, 1e6)

        mem_peak = psutil.Process().memory_info().rss / (1024 * 1024) - base_mem
        rate = num_samples / t if t > 0 else 0

        flops = len(doppler_bins) * num_samples * num_samples * 8
        gflops = flops / (t / 1000 * 1e9) if t > 0 else 0

        print(f" done ({t:.2f} ms)")
        return t, rate, gflops, mem_peak, corr

    except Exception as e:
        print(f" failed: {e}")
        return None, None, None, None, None

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_heatmap(corr_surface, doppler_bins, output_dir="results/plots"):
    """
    Generate a 2D heatmap and log-scale heatmap of the correlation surface.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale heatmap
    im1 = ax1.imshow(corr_surface, aspect='auto', origin='lower',
                     extent=[0, corr_surface.shape[1],
                             doppler_bins[0], doppler_bins[-1]],
                     cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('Code Phase (samples)')
    ax1.set_ylabel('Doppler Frequency (Hz)')
    ax1.set_title('2D CAF Correlation Surface')
    plt.colorbar(im1, ax=ax1, label='Correlation Magnitude')

    # Log-scale heatmap
    log_surface = np.log10(corr_surface + 1e-10)
    im2 = ax2.imshow(log_surface, aspect='auto', origin='lower',
                     extent=[0, corr_surface.shape[1],
                             doppler_bins[0], doppler_bins[-1]],
                     cmap='hot', interpolation='bilinear')
    ax2.set_xlabel('Code Phase (samples)')
    ax2.set_ylabel('Doppler Frequency (Hz)')
    ax2.set_title('2D CAF Surface (Log Scale)')
    plt.colorbar(im2, ax=ax2, label='Log10(Correlation)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
    plt.savefig(f"{output_dir}/correlation_heatmap.pdf", dpi=300)
    plt.close()


def plot_peak_profile(corr_surface, doppler_bins, output_dir="results/plots"):
    """
    Generate a peak profile plot showing correlation vs code phase at the best Doppler.
    """
    os.makedirs(output_dir, exist_ok=True)

    max_doppler_idx = np.argmax(np.max(corr_surface, axis=1))
    profile = corr_surface[max_doppler_idx, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(profile, 'b-', linewidth=2)
    ax.axvline(x=np.argmax(profile), color='r', linestyle='--',
               label=f'Peak at tau={np.argmax(profile)}')
    ax.set_xlabel('Code Phase (samples)')
    ax.set_ylabel('Correlation Magnitude')
    ax.set_title('CAF Profile at Best Doppler Frequency')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/peak_profile.png", dpi=300)
    plt.close()


def plot_scaling_results(df, output_dir="results/plots"):
    """
    Generate scaling plots: execution time, speedup, and real-time compliance.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Execution Time vs Sample Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Msps'], df['Sequential_Time'], 'ro-', label='Sequential')
    ax.plot(df['Msps'], df['OpenMP_Time'], 'go-', label='OpenMP')
    ax.plot(df['Msps'], df['CUDA_Time'], 'bo-', label='CUDA')
    ax.axhline(1.0, color='r', linestyle='--', label='1 ms Threshold')
    ax.set_xlabel('Sample Rate (Msps)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Execution Time vs Sample Rate')
    ax.legend()
    ax.grid(True)
    plt.savefig(f"{output_dir}/scaling_time.png", dpi=300)
    plt.close()

    # Speedup vs Sample Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Msps'], df['Speedup_Seq_CUDA'], 'bo-', label='Seq/CUDA')
    ax.plot(df['Msps'], df['Speedup_Seq_OMP'], 'go-', label='Seq/OMP')
    ax.set_xlabel('Sample Rate (Msps)')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs Sample Rate')
    ax.legend()
    ax.grid(True)
    plt.savefig(f"{output_dir}/scaling_speedup.png", dpi=300)
    plt.close()

    # Real-Time Compliance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df['Sequential_Time'], width,
           label='Sequential', color='red', alpha=0.7)
    ax.bar(x, df['OpenMP_Time'], width,
           label='OpenMP', color='green', alpha=0.7)
    ax.bar(x + width, df['CUDA_Time'], width,
           label='CUDA', color='blue', alpha=0.7)
    ax.axhline(1.0, color='black', linestyle='--',
               linewidth=2, label='1 ms Threshold')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Msps'])
    ax.set_xlabel('Sample Rate (Msps)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Real-Time Compliance')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/realtime_compliance.png", dpi=300)
    plt.close()

# =============================================================================
# SYNTHETIC TEST
# =============================================================================

def run_synthetic_test(num_samples=5000):
    """
    Execute a synthetic clean signal test on all three implementations.
    
    The test generates a clean PRN signal with minimal noise and verifies that
    each implementation produces a single sharp peak at the expected location.
    
    Args:
        num_samples: Number of samples for the test
    
    Returns:
        tuple: (passed, correlation_surface_from_cuda)
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC CLEAN SIGNAL TEST")
    print("=" * 60)
    print(f"  Samples: {num_samples}")
    print(f"  Doppler bins: {len(DOPPLER_BINS)}")
    print("  Signal: PRN-30 code with 0.01 noise (clean synthetic)")
    print("-" * 60)

    # Generate PRN code for both signal and replica
    prn_real, prn_imag = generate_prn(30, num_samples)

    # Add minimal noise to simulate real conditions
    noise_std = 0.01
    rx_r = prn_real + noise_std * np.random.randn(num_samples)
    rx_i = prn_imag + noise_std * np.random.randn(num_samples)
    rep_r, rep_i = prn_real.copy(), prn_imag.copy()

    base_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    results = []
    cuda_corr = None
    all_passed = True

    implementations = [
        ("Sequential", compute_caf_sequential),
        ("OpenMP", compute_caf_openmp),
        ("CUDA", compute_caf_cuda)
    ]

    for name, func in implementations:
        print(f"  Running {name}...", end='', flush=True)

        try:
            t, corr = time_function(func, rx_r, rx_i, rep_r, rep_i,
                                    DOPPLER_BINS, 1e6)

            # Find peak
            peak_val = np.max(corr)
            peak_loc = np.argmax(corr) % num_samples
            peak_doppler_idx = np.argmax(corr) // num_samples
            expected_tau = 0
            peak_offset = abs(peak_loc - expected_tau)
            passed = peak_offset < 10

            if not passed:
                all_passed = False

            print(f" done ({t:.2f} ms)")

            results.append({
                'Algorithm': name,
                'Time (ms)': t,
                'Peak Value': peak_val,
                'Peak Location': peak_loc,
                'Offset': peak_offset,
                'Status': 'PASS' if passed else 'FAIL'
            })

            if name == "CUDA":
                cuda_corr = corr

        except Exception as e:
            print(f" failed: {e}")
            all_passed = False
            results.append({
                'Algorithm': name,
                'Time (ms)': None,
                'Peak Value': None,
                'Peak Location': None,
                'Offset': None,
                'Status': f'ERROR: {e}'
            })

    # Print summary table
    print("\n" + "-" * 60)
    print("SYNTHETIC TEST SUMMARY")
    print("-" * 60)
    print(f"{'Algorithm':<12} {'Time (ms)':<12} {'Peak Value':<12} {'Peak tau':<10} {'Status':<10}")
    print("-" * 60)

    for r in results:
        t_str = f"{r['Time (ms)']:.2f}" if r['Time (ms)'] is not None else "N/A"
        peak_str = f"{r['Peak Value']:.4f}" if r['Peak Value'] is not None else "N/A"
        tau_str = str(r['Peak Location']) if r['Peak Location'] is not None else "N/A"
        print(f"{r['Algorithm']:<12} {t_str:<12} {peak_str:<12} {tau_str:<10} {r['Status']:<10}")

    print("-" * 60)

    if all_passed:
        print("  ALL PASSED: Synthetic test successful for all algorithms.")
        print("  The CAF implementation is working correctly.")
    else:
        print("  WARNING: One or more algorithms failed the synthetic test.")

    print("=" * 60)

    return all_passed, cuda_corr

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for the GNSS spoofing detection benchmark."""
    print("=" * 40)
    print("GNSS Spoofing Detection - PCAM CAF")
    print("=" * 40)
    print(f"Runs per test: {NUM_RUNS}")

    # Create output directories
    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    start_time = datetime.now()
    print(f"GPU: {device_name} (CC {device_cc})")
    print(f"Free GPU memory: {free_gpu_mem // (1024**2)} MB")

    # Run synthetic clean signal test
    synthetic_passed, synthetic_corr = run_synthetic_test(5000)

    # Optionally save synthetic test plots
    if synthetic_corr is not None:
        plot_heatmap(synthetic_corr, DOPPLER_BINS, "results/plots")
        plot_peak_profile(synthetic_corr, DOPPLER_BINS, "results/plots")
        print("Synthetic test plots saved to results/plots/")

    # Display menu
    print("\nMENU:")
    print("  1. Sequential")
    print("  2. OpenMP")
    print("  3. CUDA")
    print("  4. ALL (compare all on one size)")
    print("  5. Scaling Analysis (5k to 25k, all algorithms)")

    choice = int(input("Choose (1-5): "))

    doppler_bins = DOPPLER_BINS
    base_mem = psutil.Process().memory_info().rss / (1024 * 1024)

    results = []
    corr_surface = None

    # -------------------------------------------------------------------------
    # Options 1-4: Single size test
    # -------------------------------------------------------------------------
    if choice in (1, 2, 3, 4):
        num_samples = int(input("\nNumber of samples to test (0 = load entire dataset): "))

        if num_samples == 0:
            max_needed = None
        else:
            max_needed = num_samples

        # Load data
        try:
            soa_data = load_soa_data(REAL_DATA_FILE, max_needed)
            total_loaded = len(soa_data)
            print(f"Loaded {total_loaded:,} samples into SoA.")
        except FileNotFoundError:
            print("Real dataset not found. Generating synthetic data...")
            if max_needed is None:
                max_needed = 1000000
            rep_r, rep_i = generate_prn(30, max_needed)
            real = rep_r + 0.1 * np.random.randn(max_needed)
            imag = rep_i + 0.1 * np.random.randn(max_needed)
            soa_data = IQDataSoA(real=real, imag=imag)
            total_loaded = len(soa_data)
            print(f"Generated {total_loaded:,} synthetic samples.")

        if num_samples == 0:
            num_samples = total_loaded

        # Check GPU memory for CUDA
        if choice in (3, 4):
            max_safe = int(free_gpu_mem / (20 * 8))
            if num_samples > max_safe:
                print(f"Warning: {num_samples:,} samples may exceed GPU memory "
                      f"(safe limit: {max_safe:,}).")
                resp = input("Continue anyway? (y/n): ")
                if resp.lower() != 'y':
                    return

        print(f"\nBenchmarking {num_samples:,} samples...\n")

        rx_r = soa_data.real[:num_samples]
        rx_i = soa_data.imag[:num_samples]
        rep_r, rep_i = generate_prn(30, num_samples)

        if choice == 1:
            t, r, g, m, corr = run_algorithm("Sequential", compute_caf_sequential,
                                             rx_r, rx_i, rep_r, rep_i,
                                             doppler_bins, num_samples, base_mem)
            if t is not None:
                results.append(("Sequential", t, r, g, m))
                corr_surface = corr

        elif choice == 2:
            t, r, g, m, corr = run_algorithm("OpenMP", compute_caf_openmp,
                                             rx_r, rx_i, rep_r, rep_i,
                                             doppler_bins, num_samples, base_mem)
            if t is not None:
                results.append(("OpenMP", t, r, g, m))
                corr_surface = corr

        elif choice == 3:
            t, r, g, m, corr = run_algorithm("CUDA", compute_caf_cuda,
                                             rx_r, rx_i, rep_r, rep_i,
                                             doppler_bins, num_samples, base_mem)
            if t is not None:
                results.append(("CUDA", t, r, g, m))
                corr_surface = corr

        else:  # ALL
            for name, func in [("Sequential", compute_caf_sequential),
                               ("OpenMP", compute_caf_openmp),
                               ("CUDA", compute_caf_cuda)]:
                t, r, g, m, corr = run_algorithm(name, func,
                                                 rx_r, rx_i, rep_r, rep_i,
                                                 doppler_bins, num_samples,
                                                 base_mem)
                if t is not None:
                    results.append((name, t, r, g, m))
                    if name == "CUDA":
                        corr_surface = corr

        # Print single-size summary
        if results:
            print("\n" + "=" * 70)
            print("BENCHMARK SUMMARY")
            print("=" * 70)
            print(f"{'Algorithm':<12} {'Time (ms)':<12} {'Rate (samp/ms)':<15} "
                  f"{'GFLOPS':<10} {'Memory (MB)':<12}")
            print("-" * 70)

            seq_time = None
            for name, t, r, g, m in results:
                print(f"{name:<12} {t:<12.2f} {r:<15.2f} {g:<10.3f} {m:<12.2f}")
                if name == "Sequential":
                    seq_time = t

            if len(results) > 1 and seq_time:
                print("-" * 70)
                print("Speedups:")
                for name, t, r, g, m in results:
                    if name != "Sequential":
                        speedup = seq_time / t if t > 0 else 0
                        print(f"  {name} / Sequential: {speedup:.2f}x")

            print("=" * 70)

            df = pd.DataFrame(results, columns=['Algorithm', 'Time_ms',
                                                'Rate_samp_per_ms', 'GFLOPS',
                                                'Memory_MB'])
            csv_path = "results/csv/benchmark_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to {csv_path}")

        # Generate plots if correlation surface exists
        if corr_surface is not None:
            plot_heatmap(corr_surface, doppler_bins, "results/plots")
            plot_peak_profile(corr_surface, doppler_bins, "results/plots")
            print("Correlation surface plots saved to results/plots/")

    # -------------------------------------------------------------------------
    # Option 5: Scaling Analysis (5k, 10k, 15k, 20k, 25k)
    # -------------------------------------------------------------------------
    if choice == 5:
        sample_sizes = [5000, 10000, 15000, 20000, 25000]
        max_needed = max(sample_sizes)

        try:
            soa_data = load_soa_data(REAL_DATA_FILE, max_needed)
            print(f"Loaded {len(soa_data)} samples (max needed = {max_needed}).")
        except FileNotFoundError:
            print("Real dataset not found. Generating synthetic data...")
            rep_r, rep_i = generate_prn(30, max_needed)
            real = rep_r + 0.1 * np.random.randn(max_needed)
            imag = rep_i + 0.1 * np.random.randn(max_needed)
            soa_data = IQDataSoA(real=real, imag=imag)
            print(f"Generated {len(soa_data)} synthetic samples.")

        print("\n--- Scaling Analysis (5-25 Msps) ---")
        print("Sizes: 5k, 10k, 15k, 20k, 25k samples (5, 10, 15, 20, 25 Msps).")

        scaling_results = []

        for size in sample_sizes:
            if size > len(soa_data):
                print(f"Skipping {size} (exceeds loaded data).")
                continue

            print(f"\n--- Size: {size} samples ---")

            rx_r = soa_data.real[:size]
            rx_i = soa_data.imag[:size]
            rep_r, rep_i = generate_prn(30, size)

            row = {'Samples': size, 'Msps': size / 1000.0}

            for name, func in [("Sequential", compute_caf_sequential),
                               ("OpenMP", compute_caf_openmp),
                               ("CUDA", compute_caf_cuda)]:
                t, r, g, m, corr = run_algorithm(name, func,
                                                 rx_r, rx_i, rep_r, rep_i,
                                                 doppler_bins, size,
                                                 base_mem)
                if t is not None:
                    row[f'{name}_Time'] = t
                    row[f'{name}_Rate'] = r
                    row[f'{name}_GFLOPS'] = g
                    row[f'{name}_Memory'] = m

                    if name == "CUDA":
                        corr_surface = corr
                else:
                    row[f'{name}_Time'] = None

            scaling_results.append(row)

        df = pd.DataFrame(scaling_results)

        # Compute speedups
        if 'Sequential_Time' in df.columns and 'CUDA_Time' in df.columns:
            df['Speedup_Seq_OMP'] = df['Sequential_Time'] / df['OpenMP_Time']
            df['Speedup_Seq_CUDA'] = df['Sequential_Time'] / df['CUDA_Time']

        # Prepare export DataFrame with required columns
        df_export = df.copy()
        df_export.rename(columns={
            'Msps': 'SampleRate_Msps',
            'Samples': 'NumSamples',
            'Sequential_Time': 'SeqTime_ms',
            'OpenMP_Time': 'OmpTime_ms',
            'CUDA_Time': 'CudaTime_ms'
        }, inplace=True)

        df_export['NumDoppler'] = len(doppler_bins)
        df_export['Seq_GFLOPS'] = df['Sequential_GFLOPS'] if 'Sequential_GFLOPS' in df else None
        df_export['Cuda_Realtime'] = df_export['CudaTime_ms'] < 1.0
        df_export['Cuda_Bandwidth_GBs'] = (
            (3 * df_export['NumSamples'] * 8) /
            (df_export['CudaTime_ms'] / 1000 * 1e9)
        )

        # Reorder columns
        cols = ['SampleRate_Msps', 'NumSamples', 'NumDoppler',
                'SeqTime_ms', 'OmpTime_ms', 'CudaTime_ms',
                'Seq_GFLOPS', 'Cuda_Bandwidth_GBs', 'Cuda_Realtime']

        df_export = df_export[cols]

        print("\n" + "=" * 80)
        print("SCALING ANALYSIS SUMMARY")
        print("=" * 80)
        print(df_export.to_string(index=False))
        print("=" * 80)

        # Save CSV
        csv_scaling = "results/csv/scaling_analysis.csv"
        df_export.to_csv(csv_scaling, index=False)
        print(f"\nScaling results saved to {csv_scaling}")

        # Generate plots
        plot_scaling_results(df, "results/plots")

        if corr_surface is not None:
            plot_heatmap(corr_surface, doppler_bins, "results/plots")
            plot_peak_profile(corr_surface, doppler_bins, "results/plots")

        print("Plots saved to results/plots/")

    # -------------------------------------------------------------------------
    # Spoofing Detection
    # -------------------------------------------------------------------------
    if corr_surface is not None:
        result = detect_spoofing(corr_surface, doppler_bins)

        print("\n--- Spoofing Detection ---")
        status = "SPOOFING DETECTED" if result['is_spoofed'] else "AUTHENTIC SIGNAL"
        print(f"Status: {status}")
        print(f"Peak Ratio: {result['peak_ratio'] * 100:.2f}%")
        print(f"Reason: {result['reason']}")

    # -------------------------------------------------------------------------
    # Log
    # -------------------------------------------------------------------------
    log_path = f"results/logs/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_path, 'w') as f:
        f.write(f"Started: {start_time}\n")
        f.write(f"GPU: {device_name} (CC {device_cc})\n")
        f.write(f"Free GPU memory: {free_gpu_mem // (1024**2)} MB\n")
        f.write(f"Runs per test: {NUM_RUNS}\n")

        if results:
            f.write("\nBenchmark Results:\n")
            f.write(df.to_string())

        if corr_surface is not None:
            f.write("\n\nSpoofing Detection Result:\n")
            f.write(f"Status: {'Spoofed' if result['is_spoofed'] else 'Authentic'}\n")
            f.write(f"Peak Ratio: {result['peak_ratio'] * 100:.2f}%\n")

    print(f"\nLog saved to {log_path}")

    print(f"\nTotal time: {datetime.now() - start_time}")
    print("--- Done ---")


if __name__ == "__main__":
    main()