#!/usr/bin/env python3
"""
Comprehensive benchmarks for the HYDRA encryption algorithm.

This script runs benchmarks for various aspects of the HYDRA algorithm including:
- Key generation
- Encryption performance
- Decryption performance
- Memory usage
- Comparison with other algorithms (when available)
"""

import os
import sys
import time
import json
import random
import argparse
import platform
import statistics
import tracemalloc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import both the full implementation and simplified implementation
from src.core import HydraCipher
from simple_hydra import encrypt as simple_encrypt, decrypt as simple_decrypt

# Try to import other encryption libraries for comparison
HAVE_CRYPTOGRAPHY = False
HAVE_PYCRYPTODOMEX = False

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    HAVE_CRYPTOGRAPHY = True
except ImportError:
    pass

try:
    from Cryptodome.Cipher import AES
    from Cryptodome.Util.Padding import pad, unpad
    HAVE_PYCRYPTODOMEX = True
except ImportError:
    pass

class BenchmarkResult:
    """Class to store benchmark results."""
    
    def __init__(self, name):
        self.name = name
        self.data_sizes = []
        self.times = []
        self.memory = []
        self.throughput = []
    
    def add_result(self, data_size, time_taken, memory_used=None):
        """Add a benchmark result."""
        self.data_sizes.append(data_size)
        self.times.append(time_taken)
        
        # Calculate throughput in MB/s
        throughput = data_size / (time_taken * 1024 * 1024) if time_taken > 0 else 0
        self.throughput.append(throughput)
        
        if memory_used is not None:
            self.memory.append(memory_used)
    
    def average_throughput(self):
        """Get the average throughput across all data sizes."""
        if not self.throughput:
            return 0
        return statistics.mean(self.throughput)
    
    def as_dict(self):
        """Convert to a dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "data_sizes": self.data_sizes,
            "times": self.times,
            "throughput": self.throughput,
        }
        
        if self.memory:
            result["memory"] = self.memory
            
        return result

def get_system_info():
    """Get information about the system."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    
    # Try to get more CPU info on Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except:
            pass
    
    return info

def generate_random_data(size_kb):
    """Generate random data for benchmarking."""
    return os.urandom(size_kb * 1024)

def benchmark_function(func, args=(), kwargs=None, warmup=3, iterations=5):
    """Benchmark a function."""
    if kwargs is None:
        kwargs = {}
    
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Start memory tracing
    tracemalloc.start()
    
    # Benchmark runs
    times = []
    for _ in range(iterations):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "memory_peak_kb": peak / 1024,  # Convert to KB
    }

def benchmark_key_generation():
    """Benchmark key generation."""
    result = BenchmarkResult("HYDRA Key Generation")
    
    key_sizes = [256, 512]  # In bits
    iterations = 100
    
    for size_bits in key_sizes:
        size_bytes = size_bits // 8
        
        # Benchmark key generation
        def generate_key():
            return os.urandom(size_bytes)
        
        bench = benchmark_function(generate_key, iterations=iterations)
        
        print(f"{size_bits}-bit key generation: {bench['mean'] * 1000:.2f} ms")
        result.add_result(size_bytes, bench["mean"], bench["memory_peak_kb"])
    
    return result

def benchmark_hydra_full(data_sizes_kb):
    """Benchmark the full HYDRA implementation."""
    result_enc = BenchmarkResult("HYDRA Full Encryption")
    result_dec = BenchmarkResult("HYDRA Full Decryption")
    
    # Generate a key
    key = os.urandom(32)  # 256-bit key
    cipher = HydraCipher(key)
    
    for size_kb in data_sizes_kb:
        data = generate_random_data(size_kb)
        
        # Encryption benchmark
        bench_enc = benchmark_function(cipher.encrypt, args=(data,))
        print(f"HYDRA Full Encryption ({size_kb} KB): {bench_enc['mean']:.4f} s, {size_kb / bench_enc['mean'] / 1024:.2f} MB/s")
        result_enc.add_result(size_kb, bench_enc["mean"], bench_enc["memory_peak_kb"])
        
        # For decryption, we need encrypted data
        encrypted = cipher.encrypt(data)
        
        try:
            # Decryption benchmark
            bench_dec = benchmark_function(cipher.decrypt, args=(encrypted,))
            print(f"HYDRA Full Decryption ({size_kb} KB): {bench_dec['mean']:.4f} s, {size_kb / bench_dec['mean'] / 1024:.2f} MB/s")
            result_dec.add_result(size_kb, bench_dec["mean"], bench_dec["memory_peak_kb"])
        except Exception as e:
            print(f"Error in HYDRA Full Decryption ({size_kb} KB): {e}")
    
    return result_enc, result_dec

def benchmark_hydra_simple(data_sizes_kb):
    """Benchmark the simplified HYDRA implementation."""
    result_enc = BenchmarkResult("HYDRA Simple Encryption")
    result_dec = BenchmarkResult("HYDRA Simple Decryption")
    
    # Generate a key
    key = os.urandom(32)  # 256-bit key
    
    for size_kb in data_sizes_kb:
        data = generate_random_data(size_kb)
        
        # Encryption benchmark
        bench_enc = benchmark_function(simple_encrypt, args=(data, key))
        print(f"HYDRA Simple Encryption ({size_kb} KB): {bench_enc['mean']:.4f} s, {size_kb / bench_enc['mean'] / 1024:.2f} MB/s")
        result_enc.add_result(size_kb, bench_enc["mean"], bench_enc["memory_peak_kb"])
        
        # For decryption, we need encrypted data
        encrypted = simple_encrypt(data, key)
        
        # Decryption benchmark
        bench_dec = benchmark_function(simple_decrypt, args=(encrypted, key))
        print(f"HYDRA Simple Decryption ({size_kb} KB): {bench_dec['mean']:.4f} s, {size_kb / bench_dec['mean'] / 1024:.2f} MB/s")
        result_dec.add_result(size_kb, bench_dec["mean"], bench_dec["memory_peak_kb"])
    
    return result_enc, result_dec

def benchmark_aes_cryptography(data_sizes_kb):
    """Benchmark AES using cryptography library."""
    if not HAVE_CRYPTOGRAPHY:
        print("Cryptography library not available, skipping AES benchmark")
        return None, None
    
    result_enc = BenchmarkResult("AES-256-CBC (cryptography)")
    result_dec = BenchmarkResult("AES-256-CBC Decryption (cryptography)")
    
    # Generate a key and IV
    key = os.urandom(32)  # 256-bit key
    iv = os.urandom(16)  # 128-bit IV for CBC mode
    padder = padding.PKCS7(128).padder()
    unpadder = padding.PKCS7(128).unpadder()
    
    for size_kb in data_sizes_kb:
        data = generate_random_data(size_kb)
        
        # Pad the data
        padded_data = padder.update(data) + padder.finalize()
        
        # Encryption benchmark
        def encrypt_aes():
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            return encryptor.update(padded_data) + encryptor.finalize()
        
        bench_enc = benchmark_function(encrypt_aes)
        print(f"AES Encryption ({size_kb} KB): {bench_enc['mean']:.4f} s, {size_kb / bench_enc['mean'] / 1024:.2f} MB/s")
        result_enc.add_result(size_kb, bench_enc["mean"], bench_enc["memory_peak_kb"])
        
        # For decryption, we need encrypted data
        encrypted = encrypt_aes()
        
        # Decryption benchmark
        def decrypt_aes():
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()
            return unpadder.update(decrypted_padded) + unpadder.finalize()
        
        bench_dec = benchmark_function(decrypt_aes)
        print(f"AES Decryption ({size_kb} KB): {bench_dec['mean']:.4f} s, {size_kb / bench_dec['mean'] / 1024:.2f} MB/s")
        result_dec.add_result(size_kb, bench_dec["mean"], bench_dec["memory_peak_kb"])
    
    return result_enc, result_dec

def plot_results(results, title, ylabel, filename):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    
    for result in results:
        if not result.data_sizes:
            continue
        
        # Plot line with points
        sizes_mb = [size / 1024 for size in result.data_sizes]  # Convert to MB
        plt.plot(sizes_mb, result.throughput, 'o-', label=result.name)
    
    plt.title(title)
    plt.xlabel("Data Size (MB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def save_results(results, system_info, filename):
    """Save benchmark results to a JSON file."""
    output = {
        "system_info": system_info,
        "timestamp": time.time(),
        "results": [result.as_dict() for result in results]
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="HYDRA Benchmarking Tool")
    parser.add_argument("--quick", action="store_true", help="Run a quick benchmark with fewer data sizes")
    parser.add_argument("--output", default="benchmark_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Get system information
    system_info = get_system_info()
    print(f"Running benchmarks on {system_info['platform']}, {system_info.get('cpu_model', system_info['processor'])}")
    
    # Define data sizes to benchmark
    if args.quick:
        data_sizes_kb = [1, 10, 100, 1024]  # 1KB to 1MB
    else:
        data_sizes_kb = [1, 10, 100, 1024, 10*1024]  # 1KB to 10MB
    
    # Run key generation benchmark
    print("\n--- Key Generation Benchmark ---")
    key_gen_result = benchmark_key_generation()
    
    # Run HYDRA benchmarks
    print("\n--- HYDRA Full Implementation Benchmark ---")
    try:
        hydra_full_enc, hydra_full_dec = benchmark_hydra_full(data_sizes_kb)
    except Exception as e:
        print(f"Error in HYDRA Full benchmark: {e}")
        hydra_full_enc = BenchmarkResult("HYDRA Full Encryption (Failed)")
        hydra_full_dec = BenchmarkResult("HYDRA Full Decryption (Failed)")
    
    print("\n--- HYDRA Simple Implementation Benchmark ---")
    try:
        hydra_simple_enc, hydra_simple_dec = benchmark_hydra_simple(data_sizes_kb)
    except Exception as e:
        print(f"Error in HYDRA Simple benchmark: {e}")
        hydra_simple_enc = BenchmarkResult("HYDRA Simple Encryption (Failed)")
        hydra_simple_dec = BenchmarkResult("HYDRA Simple Decryption (Failed)")
    
    # Run AES benchmarks for comparison if available
    print("\n--- AES Benchmark (for comparison) ---")
    aes_enc, aes_dec = benchmark_aes_cryptography(data_sizes_kb)
    
    # Collect all results
    all_results = [
        key_gen_result,
        hydra_full_enc, hydra_full_dec,
        hydra_simple_enc, hydra_simple_dec
    ]
    
    if aes_enc and aes_dec:
        all_results.extend([aes_enc, aes_dec])
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_results(all_results, system_info, output_dir / f"hydra_benchmark_{timestamp}.json")
    
    # Create plots
    encryption_results = [r for r in all_results if "Encryption" in r.name]
    decryption_results = [r for r in all_results if "Decryption" in r.name]
    
    try:
        plot_results(
            encryption_results,
            "Encryption Performance Comparison",
            "Throughput (MB/s)",
            output_dir / f"encryption_performance_{timestamp}.png"
        )
        
        plot_results(
            decryption_results,
            "Decryption Performance Comparison",
            "Throughput (MB/s)",
            output_dir / f"decryption_performance_{timestamp}.png"
        )
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Print summary
    print("\n--- Benchmark Summary ---")
    print(f"{'Algorithm':<30} {'Avg. Encryption (MB/s)':<25} {'Avg. Decryption (MB/s)':<25}")
    print("-" * 80)
    
    for enc_result in encryption_results:
        # Find matching decryption result
        dec_name = enc_result.name.replace("Encryption", "Decryption")
        dec_result = next((r for r in decryption_results if r.name == dec_name), None)
        
        enc_throughput = enc_result.average_throughput()
        dec_throughput = dec_result.average_throughput() if dec_result else 0
        
        print(f"{enc_result.name:<30} {enc_throughput:<25.2f} {dec_throughput:<25.2f}")
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()
