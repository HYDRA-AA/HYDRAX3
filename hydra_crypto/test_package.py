#!/usr/bin/env python3
"""
Test script to verify the HYDRA crypto package functionality.

This script performs a series of tests to ensure the package is
correctly installed and functioning as expected.
"""

import os
import sys
import time
import tempfile

try:
    # Try to import from installed package
    from hydra_crypto import (
        encrypt, decrypt, generate_key, derive_key_from_password,
        encrypt_file, decrypt_file, UnifiedHydraCipher
    )
    print("✅ Successfully imported hydra_crypto package")
except ImportError:
    # If not installed, try to import from package directory
    sys.path.insert(0, os.path.abspath('.'))
    try:
        from hydra_crypto import (
            encrypt, decrypt, generate_key, derive_key_from_password,
            encrypt_file, decrypt_file, UnifiedHydraCipher
        )
        print("✅ Successfully imported from local directory")
    except ImportError as e:
        print(f"❌ Failed to import hydra_crypto package: {e}")
        print("   Make sure you run this script from the package root directory")
        print("   or install the package with: pip install -e .")
        sys.exit(1)

def run_tests():
    """Run a series of tests to verify package functionality."""
    test_count = 0
    passed = 0
    
    def run_test(name, test_func):
        nonlocal test_count, passed
        test_count += 1
        print(f"\nTest {test_count}: {name}")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            if result:
                passed += 1
                print(f"✅ Passed ({duration:.2f}s)")
                return True
            else:
                print(f"❌ Failed ({duration:.2f}s)")
                return False
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    # Test 1: Basic encryption/decryption
    def test_basic_encryption():
        key = generate_key(32)
        test_data = b"Hello, HYDRA encryption!"
        encrypted = encrypt(test_data, key)
        decrypted = decrypt(encrypted, key)
        
        print(f"  Original: {test_data}")
        print(f"  Encrypted length: {len(encrypted)} bytes")
        print(f"  Decrypted: {decrypted}")
        
        return decrypted == test_data
    
    run_test("Basic encryption/decryption", test_basic_encryption)
    
    # Test 2: Password-based key derivation
    def test_key_derivation():
        password = "test-password-123"
        key1, salt1 = derive_key_from_password(password)
        key2, _ = derive_key_from_password(password, salt1)
        key3, salt3 = derive_key_from_password(password)
        
        print(f"  Key 1: {key1.hex()[:16]}...")
        print(f"  Key 2: {key2.hex()[:16]}... (same password, same salt)")
        print(f"  Key 3: {key3.hex()[:16]}... (same password, different salt)")
        
        return key1 == key2 and key1 != key3
    
    run_test("Password-based key derivation", test_key_derivation)
    
    # Test 3: File encryption/decryption
    def test_file_encryption():
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False) as original:
            original_path = original.name
            original.write(b"This is a test file for HYDRA encryption.\n" * 100)
        
        encrypted_path = original_path + ".enc"
        decrypted_path = original_path + ".dec"
        
        try:
            # Generate a key
            key = generate_key()
            
            # Encrypt the file
            encrypt_file(original_path, encrypted_path, key)
            
            # Decrypt the file
            decrypt_file(encrypted_path, decrypted_path, key)
            
            # Compare file contents
            with open(original_path, 'rb') as f1, open(decrypted_path, 'rb') as f2:
                original_content = f1.read()
                decrypted_content = f2.read()
            
            print(f"  Original size: {len(original_content)} bytes")
            print(f"  Encrypted size: {os.path.getsize(encrypted_path)} bytes")
            print(f"  Decrypted size: {len(decrypted_content)} bytes")
            
            return original_content == decrypted_content
        finally:
            # Cleanup
            for path in [original_path, encrypted_path, decrypted_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    run_test("File encryption/decryption", test_file_encryption)
    
    # Test 4: Custom cipher parameters
    def test_custom_parameters():
        # Create a cipher with custom parameters
        key = generate_key()
        cipher = UnifiedHydraCipher(
            key=key,
            block_size=32,   # Smaller block size
            min_rounds=3,    # More minimum rounds
            max_additional_rounds=12,
            max_threads=2    # Limited threads
        )
        
        # Test data
        test_data = os.urandom(1000)  # Random 1KB data
        
        # Encrypt/decrypt with custom parameters
        encrypted = cipher.encrypt(test_data)
        decrypted = cipher.decrypt(encrypted)
        
        print(f"  Custom cipher parameters:")
        print(f"  - Block size: 32 bytes")
        print(f"  - Min rounds: 3")
        print(f"  - Max additional rounds: 12")
        print(f"  - Max threads: 2")
        print(f"  Test data size: {len(test_data)} bytes")
        print(f"  Encrypted size: {len(encrypted)} bytes")
        
        return decrypted == test_data
    
    run_test("Custom cipher parameters", test_custom_parameters)
    
    # Test 5: Parallel processing
    def test_parallel_processing():
        # Generate a larger test data set
        test_data = os.urandom(10 * 1024 * 1024)  # 10MB
        key = generate_key()
        
        # Time sequential encryption
        start_time = time.time()
        encrypted1 = encrypt(test_data, key, parallel=False)
        seq_time = time.time() - start_time
        
        # Time parallel encryption
        start_time = time.time()
        encrypted2 = encrypt(test_data, key, parallel=True)
        parallel_time = time.time() - start_time
        
        # Verify both produce valid results
        decrypted1 = decrypt(encrypted1, key)
        decrypted2 = decrypt(encrypted2, key)
        
        print(f"  Sequential time: {seq_time:.2f}s")
        print(f"  Parallel time: {parallel_time:.2f}s")
        print(f"  Speedup factor: {seq_time/parallel_time:.2f}x")
        
        return decrypted1 == decrypted2 == test_data
    
    # This test is resource-intensive, uncomment if needed
    # run_test("Parallel processing performance", test_parallel_processing)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Test Summary: {passed}/{test_count} passed")
    
    return passed == test_count

if __name__ == "__main__":
    print("HYDRA Cryptography Package Test")
    print("=" * 40)
    
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed! The package is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
