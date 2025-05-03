#!/usr/bin/env python3
"""
Benchmarks for the HYDRA encryption algorithm.
"""

import os
import time
import sys
import statistics
from typing import List, Tuple

# Add parent directory to path so we can import the src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core import HydraCipher

def benchmark_function(func, *args, iterations=10, warmup=3) -> Tuple[float, float]:
    """
    Benchmark a function by running it multiple times and measuring execution time.
    
    Args:
        func: Function to benchmark
        args: Arguments to pass to the function
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
    
    Returns:
        Tuple of (average time, standard deviation)
    """
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    
    return avg_time, std_dev

def benchmark_encryption(data_sizes: List[int], key_size=32, iterations=10) -> None:
    """
    Benchmark encryption performance for different data sizes.
    
    Args:
        data_sizes: List of data sizes to benchmark
        key_size: Key size in bytes
        iterations: Number of iterations for each benchmark
    """
    print(f"Benchmarking HYDRA encryption with {key_size * 8}-bit key")
    print(f"Running {iterations} iterations for each data size")
    print("-" * 60)
    print(f"{'Data Size (KB)':15} {'Avg Time (s)':15} {'Std Dev':15} {'Speed (MB/s)':15}")
    print("-" * 60)
    
    key = os.urandom(key_size)
    cipher = HydraCipher(key)
    
    for size in data_sizes:
        data = os.urandom(size)
        
        # Benchmark encryption
        avg_time, std_dev = benchmark_function(cipher.encrypt, data, iterations=iterations)
        
        # Calculate speed in MB/s
        speed = size / avg_time / (1024 * 1024) if avg_time > 0 else 0
        
        print(f"{size/1024:15.2f} {avg_time:15.6f} {std_dev:15.6f} {speed:15.2f}")

def benchmark_decryption(data_sizes: List[int], key_size=32, iterations=10) -> None:
    """
    Benchmark decryption performance for different data sizes.
    
    Args:
        data_sizes: List of data sizes to benchmark
        key_size: Key size in bytes
        iterations: Number of iterations for each benchmark
    """
    print(f"\nBenchmarking HYDRA decryption with {key_size * 8}-bit key")
    print(f"Running {iterations} iterations for each data size")
    print("-" * 60)
    print(f"{'Data Size (KB)':15} {'Avg Time (s)':15} {'Std Dev':15} {'Speed (MB/s)':15}")
    print("-" * 60)
    
    key = os.urandom(key_size)
    cipher = HydraCipher(key)
    
    for size in data_sizes:
        data = os.urandom(size)
        encrypted = cipher.encrypt(data)
        
        # Benchmark decryption
        avg_time, std_dev = benchmark_function(cipher.decrypt, encrypted, iterations=iterations)
        
        # Calculate speed in MB/s
        speed = size / avg_time / (1024 * 1024) if avg_time > 0 else 0
        
        print(f"{size/1024:15.2f} {avg_time:15.6f} {std_dev:15.6f} {speed:15.2f}")

def benchmark_key_setup(iterations=100) -> None:
    """
    Benchmark key setup performance.
    
    Args:
        iterations: Number of iterations for the benchmark
    """
    print(f"\nBenchmarking HYDRA key setup")
    print(f"Running {iterations} iterations")
    print("-" * 60)
    print(f"{'Key Size (bits)':15} {'Avg Time (s)':15} {'Std Dev':15}")
    print("-" * 60)
    
    for key_size in [32, 64]:  # 256-bit and 512-bit keys
        def setup_key():
            key = os.urandom(key_size)
            HydraCipher(key)
        
        # Benchmark key setup
        avg_time, std_dev = benchmark_function(setup_key, iterations=iterations)
        
        print(f"{key_size * 8:15} {avg_time:15.6f} {std_dev:15.6f}")

def main():
    """Main function."""
    # Data sizes to benchmark (in bytes)
    data_sizes = [
        1 * 1024,       # 1 KB
        10 * 1024,      # 10 KB
        100 * 1024,     # 100 KB
        1024 * 1024,    # 1 MB
        # Uncomment for larger benchmarks (may take a while)
        # 10 * 1024 * 1024,  # 10 MB
    ]
    
    # Run benchmarks
    benchmark_encryption(data_sizes)
    benchmark_decryption(data_sizes)
    benchmark_key_setup()

if __name__ == '__main__':
    main()
