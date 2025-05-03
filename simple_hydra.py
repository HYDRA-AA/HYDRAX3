#!/usr/bin/env python3
"""
Simplified HYDRA Encryption for testing.
"""

import os
import hashlib
import struct
import binascii

def xor_bytes(a, b):
    """XOR two byte arrays."""
    return bytes(x ^ y for x, y in zip(a, b))

def simple_encrypt(plaintext, key):
    """
    Simple encryption: just XOR with key-derived values.
    
    Args:
        plaintext: 64-byte block to encrypt
        key: 32-byte or 64-byte key
    
    Returns:
        64-byte encrypted block
    """
    # Derive round keys from the master key (just a simple hash)
    h = hashlib.sha512()
    h.update(key)
    h.update(b"round_key")
    round_key = h.digest()
    
    # Simple XOR encryption
    ciphertext = xor_bytes(plaintext, round_key)
    return ciphertext

def simple_decrypt(ciphertext, key):
    """
    Simple decryption: just XOR with key-derived values.
    
    Args:
        ciphertext: 64-byte block to decrypt
        key: 32-byte or 64-byte key
    
    Returns:
        64-byte decrypted block
    """
    # Derive round keys from the master key (same as in encrypt)
    h = hashlib.sha512()
    h.update(key)
    h.update(b"round_key")
    round_key = h.digest()
    
    # Simple XOR decryption
    plaintext = xor_bytes(ciphertext, round_key)
    return plaintext

def pad_message(message):
    """
    Pad the message to a multiple of 64 bytes.
    
    Args:
        message: Message to pad
    
    Returns:
        Padded message
    """
    # Calculate padding length
    padded_length = ((len(message) + 63) // 64) * 64
    padding_length = padded_length - len(message)
    
    # Add padding
    padded_message = bytearray(message) + bytearray([padding_length] * padding_length)
    return padded_message

def unpad_message(padded_message):
    """
    Remove padding from the message.
    
    Args:
        padded_message: Padded message
    
    Returns:
        Original message
    """
    # Get padding length from the last byte
    padding_length = padded_message[-1]
    
    # Verify padding
    if padding_length > 64:
        raise ValueError(f"Invalid padding length: {padding_length}")
    
    # Verify padding bytes
    for i in range(1, padding_length + 1):
        if padded_message[-i] != padding_length:
            raise ValueError(f"Invalid padding byte at position -{i}")
    
    # Remove padding
    return padded_message[:-padding_length]

def encrypt(message, key):
    """
    Encrypt a message.
    
    Args:
        message: Message to encrypt
        key: 32-byte or 64-byte key
    
    Returns:
        Encrypted message
    """
    # Pad the message
    padded_message = pad_message(message)
    
    # Encrypt each block
    ciphertext = bytearray()
    for i in range(0, len(padded_message), 64):
        block = padded_message[i:i+64]
        encrypted_block = simple_encrypt(block, key)
        ciphertext.extend(encrypted_block)
    
    return bytes(ciphertext)

def decrypt(ciphertext, key):
    """
    Decrypt a message.
    
    Args:
        ciphertext: Message to decrypt
        key: 32-byte or 64-byte key
    
    Returns:
        Decrypted message
    """
    # Make sure the ciphertext is a multiple of 64 bytes
    if len(ciphertext) % 64 != 0:
        raise ValueError(f"Ciphertext length must be a multiple of 64 bytes (got {len(ciphertext)})")
    
    # Decrypt each block
    plaintext = bytearray()
    for i in range(0, len(ciphertext), 64):
        block = ciphertext[i:i+64]
        decrypted_block = simple_decrypt(block, key)
        plaintext.extend(decrypted_block)
    
    # Remove padding
    return unpad_message(plaintext)

def main():
    # Create a fixed key
    key = bytes.fromhex("00112233445566778899aabbccddeeff" * 2)
    print(f"Key: {binascii.hexlify(key).decode()}")
    
    # Test message
    message = b"Hello, this is a test of the simplified HYDRA encryption!"
    print(f"Original message: '{message.decode()}'")
    print(f"Length: {len(message)} bytes")
    
    # Encrypt
    ciphertext = encrypt(message, key)
    print(f"\nEncrypted: {binascii.hexlify(ciphertext).decode()}")
    print(f"Length: {len(ciphertext)} bytes")
    
    # Show padding details
    padded = pad_message(message)
    print(f"\nPadding details:")
    print(f"Padded length: {len(padded)} bytes")
    print(f"Padding length: {padded[-1]}")
    print(f"Last few bytes: {binascii.hexlify(padded[-10:]).decode()}")
    
    # Decrypt
    try:
        decrypted = decrypt(ciphertext, key)
        print(f"\nDecrypted: '{decrypted.decode()}'")
        print(f"Length: {len(decrypted)} bytes")
        
        # Verify
        if decrypted == message:
            print("\n✓ Success: Decrypted message matches original")
        else:
            print("\n✗ Error: Decrypted message does not match original")
    except Exception as e:
        print(f"\n✗ Decryption error: {e}")
        
        # Debug information
        print("\nDebug information:")
        plaintext = bytearray()
        for i in range(0, len(ciphertext), 64):
            block = ciphertext[i:i+64]
            decrypted_block = simple_decrypt(block, key)
            plaintext.extend(decrypted_block)
        
        print(f"Raw decrypted length: {len(plaintext)} bytes")
        print(f"Last byte: {plaintext[-1]}")

if __name__ == "__main__":
    main()
