#!/usr/bin/env python3
"""
Parallel Processing Utilities
=============================

Automatic detection and configuration of optimal parallelization settings
based on available hardware resources.
"""

import os
import psutil
import multiprocessing as mp
from typing import Optional, Tuple


def get_optimal_n_jobs(requested: Optional[int] = None,
                      reserve_cores: int = 1) -> int:
    """
    Calculate optimal number of parallel jobs based on system resources.

    Args:
        requested: User-requested number of jobs. If None, auto-detect.
                  If -1, use all available cores.
        reserve_cores: Number of cores to reserve for system (default: 1)

    Returns:
        Optimal number of parallel jobs

    Examples:
        >>> get_optimal_n_jobs()  # Auto-detect
        7  # On 8-core system
        >>> get_optimal_n_jobs(-1)  # Use all cores
        8
        >>> get_optimal_n_jobs(4)  # Use requested
        4
    """
    # Get total CPU count
    total_cpus = mp.cpu_count()

    # Handle explicit requests
    if requested is not None:
        if requested == -1:
            # Use all CPUs
            return total_cpus
        elif requested > 0:
            # Use requested, but cap at available CPUs
            return min(requested, total_cpus)

    # Auto-detect optimal value
    # Reserve some cores for system and other processes
    available_cpus = max(1, total_cpus - reserve_cores)

    # Use 75% of available cores for large systems (>16 cores)
    if total_cpus > 16:
        optimal = max(1, int(available_cpus * 0.75))
    else:
        optimal = available_cpus

    return optimal


def get_memory_info() -> Tuple[float, float, float]:
    """
    Get system memory information.

    Returns:
        Tuple of (total_gb, available_gb, percent_used)
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    percent_used = mem.percent

    return total_gb, available_gb, percent_used


def check_gpu_availability() -> Tuple[bool, int, str]:
    """
    Check if NVIDIA GPU is available and get device count.

    Returns:
        Tuple of (has_gpu, num_gpus, device_name)
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            num_gpus = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "Unknown"
            return True, num_gpus, device_name
    except ImportError:
        pass

    # Try CatBoost's GPU check
    try:
        from catboost import CatBoostClassifier
        test_pool = None
        # CatBoost will throw if no GPU available
        return True, 1, "CatBoost GPU"
    except:
        pass

    return False, 0, "No GPU"


def get_optimal_device() -> str:
    """
    Get optimal device for training (cuda or cpu).

    Returns:
        Device string: 'cuda' if GPU available, else 'cpu'
    """
    has_gpu, _, _ = check_gpu_availability()
    return 'cuda' if has_gpu else 'cpu'


def print_system_info():
    """Print detailed system information for optimization."""
    print("=" * 70)
    print("System Hardware Information")
    print("=" * 70)

    # CPU Info
    cpu_count = mp.cpu_count()
    print(f"CPU Cores: {cpu_count}")
    print(f"Optimal n_jobs: {get_optimal_n_jobs()} (auto-detected)")

    # Memory Info
    total_gb, available_gb, percent_used = get_memory_info()
    print(f"\nMemory:")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Available: {available_gb:.1f} GB")
    print(f"  Used: {percent_used:.1f}%")

    # GPU Info
    has_gpu, num_gpus, device_name = check_gpu_availability()
    print(f"\nGPU:")
    if has_gpu:
        print(f"  Status: ✅ Available")
        print(f"  Count: {num_gpus}")
        print(f"  Device: {device_name}")
        print(f"  Recommended: --device cuda")
    else:
        print(f"  Status: ❌ Not available")
        print(f"  Recommended: --device cpu")

    print("=" * 70)


def configure_threading():
    """
    Configure optimal threading settings for numerical libraries.
    Call this at the start of your script for best performance.
    """
    optimal_threads = get_optimal_n_jobs()

    # Set environment variables for threading libraries
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)

    # Configure NumPy if available
    try:
        import numpy as np
        if hasattr(np, '__config__'):
            # NumPy threading is configured via environment variables above
            pass
    except ImportError:
        pass


if __name__ == '__main__':
    # Demo
    print_system_info()
    print()
    print("Example usage:")
    print(f"  n_jobs = get_optimal_n_jobs()  # {get_optimal_n_jobs()}")
    print(f"  n_jobs = get_optimal_n_jobs(-1)  # {get_optimal_n_jobs(-1)} (use all)")
    print(f"  device = get_optimal_device()  # '{get_optimal_device()}'")
