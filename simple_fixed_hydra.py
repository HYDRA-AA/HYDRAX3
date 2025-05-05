#!/usr/bin/env python3
"""
A simple fixed version of HYDRA that correctly handles adaptive rounds.

This implementation takes a direct approach to encrypt/decrypt and store metadata.
"""

import os
import hashlib

def xor_bytes(a: bytes, b: bytes) -> bytes:
    """Simple XOR of two byte arrays."""
    return bytes(x ^ y for x, y in zip(a, b))

def pad_data(data: bytes) -> bytes:
    """Apply PKCS#7 padding."""
    block_size = 16  # Using a smaller block size for simplicity
    padding_len = block_size - (len(data) % block_size)
    if padding_len == 0:
        padding_len = block_size
    padding = bytes([padding_len]) * padding_len
    return data + padding

def unpad_data(data: bytes) -> bytes:
    """Remove PKCS#7 padding."""
    padding_len = data[-1]
    if padding_len == 0 or padding_len > 16:
        raise ValueError(f"Invalid padding length: {padding_len}")
    for i in range(1, padding_len + 1):
        if data[-i] != padding_len:
            raise ValueError(f"Invalid padding byte at position {-i}")
    return data[:-padding_len]

def derive_round_key(key: bytes, counter: int) -> bytes:
    """Derive a round key using the main key and a counter."""
    h = hashlib.sha256()
    h.update(key)
    h.update(counter.to_bytes(4, byteorder='little'))
    return h.digest()

def simple_encrypt(data: bytes, key: bytes) -> bytes:
    """
    Encrypt data using a simple block cipher with adaptive rounds.
    
    Args:
        data: Data to encrypt
        key: Secret key
        
    Returns:
        Encrypted data with metadata
    """
    # Pad the data
    padded_data = pad_data(data)
    print(f"Padded data length: {len(padded_data)} bytes")
    
    # Prepare result buffer
    result = bytearray()
    
    # Process each block
    block_size = 16
    for i in range(0, len(padded_data), block_size):
        block = padded_data[i:i+block_size]
        
        # Calculate adaptive rounds based on block complexity
        unique_bytes = len(set(block))
        complexity = unique_bytes / 256.0
        additional_rounds = int(complexity * 4)  # 0-4 additional rounds
        min_rounds = 1
        
        # Store the metadata: number of rounds
        result.append(additional_rounds)
        
        # Encrypt the block
        encrypted_block = bytearray(block)
        total_rounds = min_rounds + additional_rounds
        
        # Apply rounds
        for r in range(total_rounds):
            # Get round key
            round_key = derive_round_key(key, r)
            # XOR with round key (simple encryption)
            for j in range(len(encrypted_block)):
                encrypted_block[j] ^= round_key[j % len(round_key)]
        
        # Add encrypted block to result
        result.extend(encrypted_block)
    
    print(f"Final encrypted length: {len(result)} bytes")
    return bytes(result)

def simple_decrypt(encrypted_data: bytes, key: bytes) -> bytes:
    """
    Decrypt data using the algorithm with stored metadata.
    
    Args:
        encrypted_data: Encrypted data with metadata
        key: Secret key
        
    Returns:
        Decrypted data
    """
    # Prepare result buffer
    result = bytearray()
    
    # Process each block with its metadata
    block_size = 16
    pos = 0
    min_rounds = 1
    
    while pos < len(encrypted_data):
        # Read metadata
        if pos >= len(encrypted_data):
            break
        additional_rounds = encrypted_data[pos]
        pos += 1
        
        # Read encrypted block
        if pos + block_size > len(encrypted_data):
            print(f"Warning: Incomplete block at {pos}, need {block_size} more bytes")
            break
        encrypted_block = encrypted_data[pos:pos+block_size]
        pos += block_size
        
        # Decrypt the block using the stored round count
        decrypted_block = bytearray(encrypted_block)
        total_rounds = min_rounds + additional_rounds
        
        # Apply rounds in reverse
        for r in range(total_rounds - 1, -1, -1):
            # Get round key (same as encryption)
            round_key = derive_round_key(key, r)
            # XOR with round key (simple decryption is the same as encryption for XOR)
            for j in range(len(decrypted_block)):
                decrypted_block[j] ^= round_key[j % len(round_key)]
        
        # Add decrypted block to result
        result.extend(decrypted_block)
    
    # Remove padding
    try:
        unpadded = unpad_data(result)
        return unpadded
    except ValueError as e:
        print(f"Warning: {e}")
        return bytes(result)

def test_encryption(test_data=None):
    """
    Test the encryption/decryption with adaptive rounds.
    """
    # Generate a random key
    key = os.urandom(32)
    print(f"Using key: {key.hex()[:16]}...")
    
    # Test data
    if not test_data:
        test_data = b"This is a test of the fixed HYDRA encryption system with metadata."
    print(f"Original data: {test_data}")
    print(f"Original length: {len(test_data)} bytes")
    
    # Encrypt
    encrypted = simple_encrypt(test_data, key)
    print(f"Encrypted data length: {len(encrypted)} bytes")
    
    # Decrypt
    decrypted = simple_decrypt(encrypted, key)
    print(f"Decrypted data: {decrypted}")
    print(f"Decrypted length: {len(decrypted)} bytes")
    
    # Verify
    if decrypted == test_data:
        print("SUCCESS: Decryption matches original data!")
    else:
        print("ERROR: Decryption does not match original data!")
        # Compare byte by byte for debugging
        min_len = min(len(test_data), len(decrypted))
        for i in range(min_len):
            if test_data[i] != decrypted[i]:
                print(f"First difference at position {i}: {test_data[i]} (0x{test_data[i]:02x}) != {decrypted[i]} (0x{decrypted[i]:02x})")
                break

if __name__ == "__main__":
    test_encryption()
    
    # Also test with random data
    print("\nTesting with random data:")
    random_data = os.urandom(100)
    test_encryption(random_data)
