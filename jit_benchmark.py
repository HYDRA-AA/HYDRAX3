#!/usr/bin/env python3
"""
Benchmark comparing the JIT-accelerated HYDRA implementation with the original.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Import both implementations
import jit_hydra
from src.core.hydra_cipher import HydraCipher

def run_benchmark(data_sizes_kb: List[int], iterations: int = 3) -> Dict[str, Any]:
    """
    Run a benchmark comparing original HydraCipher vs JIT-accelerated JitHydraCipher.
    
    Args:
        data_sizes_kb: List of data sizes to test in KB
        iterations: Number of iterations for each test
        
    Returns:
        Dictionary containing benchmark results
    """
    results = {
        "sizes": data_sizes_kb,
        "original_encrypt": [],
        "original_decrypt": [],
        "jit_encrypt": [],
        "jit_decrypt": [],
        "jit_threaded_encrypt": [],
        "jit_threaded_decrypt": []
    }
    
    # Generate a fixed key for consistency
    key = os.urandom(32)
    
    # Initialize ciphers
    original_cipher = HydraCipher(key)
    jit_cipher = jit_hydra.JitHydraCipher(key, max_threads=1)  # Single-threaded for fair comparison
    jit_threaded_cipher = jit_hydra.JitHydraCipher(key)  # Default multi-threading
    
    for size_kb in data_sizes_kb:
        print(f"Testing with data size: {size_kb} KB")
        
        # Generate test data - repeatable pattern
        chunk = b"HYDRA Encryption Algorithm Benchmark " * 25  # ~1KB chunk
        repeats = size_kb
        data = chunk * repeats

        # Benchmark: Original encrypt
        original_encrypt_times = []
        for _ in range(iterations):
            start_time = time.time()
            encrypted_original = original_cipher.encrypt(data)
            end_time = time.time()
            original_encrypt_times.append(end_time - start_time)
        avg_original_encrypt = sum(original_encrypt_times) / iterations
        results["original_encrypt"].append(avg_original_encrypt)
        
        # Benchmark: Original decrypt
        original_decrypt_times = []
        for _ in range(iterations):
            start_time = time.time()
            decrypted_original = original_cipher.decrypt(encrypted_original)
            end_time = time.time()
            original_decrypt_times.append(end_time - start_time)
        avg_original_decrypt = sum(original_decrypt_times) / iterations
        results["original_decrypt"].append(avg_original_decrypt)
        
        # Check correctness
        if decrypted_original != data:
            print(f"⚠️ WARNING: Original cipher decryption mismatch at {size_kb} KB")
        else:
            print(f"✓ Original cipher: Verified successful encryption/decryption at {size_kb} KB")
        
        # JIT Cipher (Single-threaded)
        
        # Benchmark: JIT encrypt
        jit_encrypt_times = []
        for _ in range(iterations):
            start_time = time.time()
            encrypted_jit = jit_cipher.encrypt(data)
            end_time = time.time()
            jit_encrypt_times.append(end_time - start_time)
        avg_jit_encrypt = sum(jit_encrypt_times) / iterations
        results["jit_encrypt"].append(avg_jit_encrypt)
        
        # Benchmark: JIT decrypt
        jit_decrypt_times = []
        for _ in range(iterations):
            start_time = time.time()
            decrypted_jit = jit_cipher.decrypt(encrypted_jit)
            end_time = time.time()
            jit_decrypt_times.append(end_time - start_time)
        avg_jit_decrypt = sum(jit_decrypt_times) / iterations
        results["jit_decrypt"].append(avg_jit_decrypt)
        
        # Check correctness
        if decrypted_jit != data:
            print(f"⚠️ WARNING: JIT cipher decryption mismatch at {size_kb} KB")
        else:
            print(f"✓ JIT cipher: Verified successful encryption/decryption at {size_kb} KB")
        
        # Multi-threaded JIT Cipher (only for larger sizes)
        if size_kb >= 8:  # Only use threading for larger sizes
            # Benchmark: JIT threaded encrypt
            jit_threaded_encrypt_times = []
            for _ in range(iterations):
                start_time = time.time()
                encrypted_jit_threaded = jit_threaded_cipher.encrypt(data)
                end_time = time.time()
                jit_threaded_encrypt_times.append(end_time - start_time)
            avg_jit_threaded_encrypt = sum(jit_threaded_encrypt_times) / iterations
            results["jit_threaded_encrypt"].append(avg_jit_threaded_encrypt)
            
            # Benchmark: JIT threaded decrypt
            jit_threaded_decrypt_times = []
            for _ in range(iterations):
                start_time = time.time()
                decrypted_jit_threaded = jit_threaded_cipher.decrypt(encrypted_jit_threaded)
                end_time = time.time()
                jit_threaded_decrypt_times.append(end_time - start_time)
            avg_jit_threaded_decrypt = sum(jit_threaded_decrypt_times) / iterations
            results["jit_threaded_decrypt"].append(avg_jit_threaded_decrypt)
            
            # Check correctness
            if decrypted_jit_threaded != data:
                print(f"⚠️ WARNING: Threaded JIT cipher decryption mismatch at {size_kb} KB")
            else:
                print(f"✓ Threaded JIT cipher: Verified successful encryption/decryption at {size_kb} KB")
        else:
            # Placeholder for small sizes where threading isn't used
            results["jit_threaded_encrypt"].append(avg_jit_encrypt)
            results["jit_threaded_decrypt"].append(avg_jit_decrypt)
    
    return results

def calculate_speedups(results: Dict[str, Any]) -> Dict[str, List[float]]:
    """Calculate speedup factors compared to original implementation."""
    speedups = {
        "jit_encrypt": [],
        "jit_decrypt": [],
        "jit_threaded_encrypt": [],
        "jit_threaded_decrypt": []
    }
    
    for i in range(len(results["sizes"])):
        # Avoid division by zero
        orig_encrypt = max(results["original_encrypt"][i], 1e-10)
        orig_decrypt = max(results["original_decrypt"][i], 1e-10)
        
        speedups["jit_encrypt"].append(orig_encrypt / results["jit_encrypt"][i])
        speedups["jit_decrypt"].append(orig_decrypt / results["jit_decrypt"][i])
        speedups["jit_threaded_encrypt"].append(orig_encrypt / results["jit_threaded_encrypt"][i])
        speedups["jit_threaded_decrypt"].append(orig_decrypt / results["jit_threaded_decrypt"][i])
    
    return speedups

def calculate_throughputs(results: Dict[str, Any]) -> Dict[str, List[float]]:
    """Calculate throughput in MB/s."""
    throughputs = {
        "original_encrypt": [],
        "original_decrypt": [],
        "jit_encrypt": [],
        "jit_decrypt": [],
        "jit_threaded_encrypt": [],
        "jit_threaded_decrypt": []
    }
    
    for i, size_kb in enumerate(results["sizes"]):
        size_mb = size_kb / 1024.0
        
        # Avoid division by zero
        orig_encrypt = max(results["original_encrypt"][i], 1e-10)
        orig_decrypt = max(results["original_decrypt"][i], 1e-10)
        jit_encrypt = max(results["jit_encrypt"][i], 1e-10)
        jit_decrypt = max(results["jit_decrypt"][i], 1e-10)
        jit_tr_encrypt = max(results["jit_threaded_encrypt"][i], 1e-10)
        jit_tr_decrypt = max(results["jit_threaded_decrypt"][i], 1e-10)
        
        throughputs["original_encrypt"].append(size_mb / orig_encrypt)
        throughputs["original_decrypt"].append(size_mb / orig_decrypt)
        throughputs["jit_encrypt"].append(size_mb / jit_encrypt)
        throughputs["jit_decrypt"].append(size_mb / jit_decrypt)
        throughputs["jit_threaded_encrypt"].append(size_mb / jit_tr_encrypt)
        throughputs["jit_threaded_decrypt"].append(size_mb / jit_tr_decrypt)
    
    return throughputs

def print_results(results: Dict[str, Any], speedups: Dict[str, List[float]], throughputs: Dict[str, List[float]]):
    """Print benchmark results in a table format."""
    print("\n" + "=" * 90)
    print(f"{'Size (KB)':<10} | {'Original (s)':<15} | {'JIT (s)':<15} | {'JIT-MT (s)':<15} | {'Speedup':<10} | {'MT Speedup':<10}")
    print("=" * 90)
    
    for i, size in enumerate(results["sizes"]):
        # Encryption
        print(f"{size:<10} | "
              f"{results['original_encrypt'][i]:<15.6f} | "
              f"{results['jit_encrypt'][i]:<15.6f} | "
              f"{results['jit_threaded_encrypt'][i]:<15.6f} | "
              f"{speedups['jit_encrypt'][i]:<10.2f}x | "
              f"{speedups['jit_threaded_encrypt'][i]:<10.2f}x | "
              f"[Encryption]")
        
        # Decryption
        print(f"{size:<10} | "
              f"{results['original_decrypt'][i]:<15.6f} | "
              f"{results['jit_decrypt'][i]:<15.6f} | "
              f"{results['jit_threaded_decrypt'][i]:<15.6f} | "
              f"{speedups['jit_decrypt'][i]:<10.2f}x | "
              f"{speedups['jit_threaded_decrypt'][i]:<10.2f}x | "
              f"[Decryption]")
        
        print("-" * 90)
    
    print("\nThroughput (MB/s):")
    print("=" * 90)
    print(f"{'Size (KB)':<10} | {'Original Enc':<15} | {'JIT Enc':<15} | {'JIT-MT Enc':<15} | {'Original Dec':<15} | {'JIT Dec':<15} | {'JIT-MT Dec':<15}")
    print("=" * 90)
    
    for i, size in enumerate(results["sizes"]):
        print(f"{size:<10} | "
              f"{throughputs['original_encrypt'][i]:<15.2f} | "
              f"{throughputs['jit_encrypt'][i]:<15.2f} | "
              f"{throughputs['jit_threaded_encrypt'][i]:<15.2f} | "
              f"{throughputs['original_decrypt'][i]:<15.2f} | "
              f"{throughputs['jit_decrypt'][i]:<15.2f} | "
              f"{throughputs['jit_threaded_decrypt'][i]:<15.2f}")
    
    print("=" * 90)

def plot_results(results: Dict[str, Any], throughputs: Dict[str, List[float]], filename: str = "hydra_benchmark.png"):
    """Create and save a plot comparing the different implementations."""
    plt.figure(figsize=(12, 10))
    
    # Throughput subplot
    plt.subplot(2, 1, 1)
    plt.title('HYDRA Performance Comparison: Throughput')
    plt.plot(results["sizes"], throughputs["original_encrypt"], 'bo-', label='Original (Encrypt)')
    plt.plot(results["sizes"], throughputs["original_decrypt"], 'b*--', label='Original (Decrypt)')
    plt.plot(results["sizes"], throughputs["jit_encrypt"], 'ro-', label='JIT (Encrypt)')
    plt.plot(results["sizes"], throughputs["jit_decrypt"], 'r*--', label='JIT (Decrypt)')
    plt.plot(results["sizes"], throughputs["jit_threaded_encrypt"], 'go-', label='JIT+MT (Encrypt)')
    plt.plot(results["sizes"], throughputs["jit_threaded_decrypt"], 'g*--', label='JIT+MT (Decrypt)')
    plt.xlabel('Data Size (KB)')
    plt.ylabel('Throughput (MB/s)')
    plt.grid(True)
    plt.legend()
    
    # Speedup subplot
    speedups = calculate_speedups(results)
    plt.subplot(2, 1, 2)
    plt.title('HYDRA Performance Comparison: Speedup Factor')
    plt.plot(results["sizes"], speedups["jit_encrypt"], 'ro-', label='JIT Speedup (Encrypt)')
    plt.plot(results["sizes"], speedups["jit_decrypt"], 'r*--', label='JIT Speedup (Decrypt)')
    plt.plot(results["sizes"], speedups["jit_threaded_encrypt"], 'go-', label='JIT+MT Speedup (Encrypt)')
    plt.plot(results["sizes"], speedups["jit_threaded_decrypt"], 'g*--', label='JIT+MT Speedup (Decrypt)')
    plt.xlabel('Data Size (KB)')
    plt.ylabel('Speedup Factor (x)')
    plt.grid(True)
    plt.axhline(y=1.0, color='k', linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Performance plot saved as '{filename}'")

def main():
    # Test with different data sizes
    data_sizes = [1, 4, 16, 64, 256]
    iterations = 3
    
    print(f"Running benchmarks with {iterations} iterations for each data size...")
    results = run_benchmark(data_sizes, iterations)
    
    # Calculate speedups and throughputs
    speedups = calculate_speedups(results)
    throughputs = calculate_throughputs(results)
    
    # Print results
    print_results(results, speedups, throughputs)
    
    # Plot results (if matplotlib is available)
    try:
        plot_results(results, throughputs)
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    # Print summary
    avg_jit_speedup = sum(speedups["jit_encrypt"] + speedups["jit_decrypt"]) / (2 * len(results["sizes"]))
    avg_threaded_speedup = sum(speedups["jit_threaded_encrypt"] + speedups["jit_threaded_decrypt"]) / (2 * len(results["sizes"]))
    
    print("\nPerformance Summary:")
    print(f"- JIT Implementation: Average {avg_jit_speedup:.1f}x speedup")
    print(f"- JIT with Threading: Average {avg_threaded_speedup:.1f}x speedup")

if __name__ == "__main__":
    main()
