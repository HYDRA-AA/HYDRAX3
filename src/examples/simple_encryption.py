#!/usr/bin/env python3
"""
HYDRA Encryption Algorithm - Simple Usage Example

This example demonstrates basic usage of the HYDRA encryption algorithm.
"""

import os
import sys
import time
import binascii

# Add the parent directory to the path so we can import the core package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core import HydraCipher

def generate_random_key(size=32):
    """Generate a random key of the specified size."""
    return os.urandom(size)

def print_hex(label, data):
    """Print data in hexadecimal format."""
    hex_data = binascii.hexlify(data).decode('ascii')
    print(f"{label}: {hex_data}")

def main():
    # Generate a random 256-bit key (32 bytes)
    key = generate_random_key(32)
    print_hex("Key", key)
    
    # Create a cipher instance
    cipher = HydraCipher(key)
    
    # Message to encrypt
    message = b"This is a secret message encrypted with HYDRA, a novel encryption algorithm designed for high security against both classical and quantum attacks."
    print(f"Original message: {message.decode('utf-8')}")
    
    # Encrypt the message
    print("\nEncrypting...")
    start_time = time.time()
    encrypted = cipher.encrypt(message)
    encrypt_time = time.time() - start_time
    print_hex("Encrypted", encrypted)
    print(f"Encryption took {encrypt_time:.6f} seconds")
    
    # Decrypt the message
    print("\nDecrypting...")
    start_time = time.time()
    try:
        # Print some debug info about the encrypted data
        print(f"Encrypted length: {len(encrypted)} bytes")
        print(f"Last byte value: {encrypted[-1]}")
        
        # Try decryption with error handling
        decrypted = cipher.decrypt(encrypted)
        decrypt_time = time.time() - start_time
        
        print(f"Decryption succeeded!")
        print(f"Decrypted length: {len(decrypted)} bytes")
        print(f"Decrypted message: {decrypted.decode('utf-8')}")
        print(f"Decryption took {decrypt_time:.6f} seconds")
        
        # Verify the decryption
        if message == decrypted:
            print("\nSuccessful encryption and decryption!")
        else:
            print("\nWARNING: Decrypted message does not match original message!")
            print(f"Original length: {len(message)}")
            print(f"Decrypted length: {len(decrypted)}")
            # Find where they differ
            for i in range(min(len(message), len(decrypted))):
                if message[i] != decrypted[i]:
                    print(f"First difference at position {i}:")
                    print(f"  Original: {message[i:i+10]}")
                    print(f"  Decrypted: {decrypted[i:i+10]}")
                    break
    except Exception as e:
        print(f"Decryption error: {e}")
        print("Trying a workaround approach...")
        
        # Direct block-by-block decryption as a workaround
        plaintext = bytearray()
        for i in range(0, len(encrypted), 64):
            block = encrypted[i:i+64]
            try:
                decrypted_block = cipher.decrypt_block(block)
                plaintext.extend(decrypted_block)
            except Exception as e:
                print(f"Error decrypting block at position {i}: {e}")
        
        # Try to recover the message
        try:
            # Look for valid text in the output
            recovered_text = plaintext.decode('utf-8', errors='replace')
            print(f"\nPartially recovered text: {recovered_text[:100]}...")
        except Exception as e:
            print(f"Failed to recover text: {e}")

if __name__ == "__main__":
    main()
