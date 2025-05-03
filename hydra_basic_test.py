#!/usr/bin/env python3
"""
Basic test for HYDRA encryption to debug the padding issue.
"""

import os
import sys
import binascii

# Add the project to the path
sys.path.append(os.path.abspath("."))

from src.core import HydraCipher

def main():
    print("HYDRA Basic Encryption Test")
    print("--------------------------\n")
    
    # Create a fixed key for reproducibility
    key = bytes.fromhex("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
    print(f"Using fixed key: {binascii.hexlify(key).decode()}")
    
    # Create cipher instance
    cipher = HydraCipher(key)
    
    # Test with a very simple message
    message = b"Hello, HYDRA!"
    print(f"Original message: '{message.decode()}'")
    print(f"Original message length: {len(message)} bytes")
    
    # Encrypt
    print("\nEncrypting...")
    encrypted = cipher.encrypt(message)
    print(f"Encrypted data (hex): {binascii.hexlify(encrypted).decode()}")
    print(f"Encrypted length: {len(encrypted)} bytes")
    
    # Debug: Examine the padded plaintext
    padded_length = ((len(message) + 63) // 64) * 64
    padding_length = padded_length - len(message)
    padded_message = bytearray(message) + bytearray([padding_length] * padding_length)
    print(f"\nDebugging:")
    print(f"Padded message length: {len(padded_message)} bytes")
    print(f"Padding length: {padding_length}")
    print(f"Padding value (last byte): {padded_message[-1]}")
    
    # Manually check block sizes
    for i in range(0, len(padded_message), 64):
        block = padded_message[i:i+64]
        print(f"Block {i//64} length: {len(block)} bytes")
    
    # Try decryption with manual padding check
    print(f"\nDecrypting...")
    try:
        decrypted = cipher.decrypt(encrypted)
        print(f"Decryption successful!")
        print(f"Decrypted message: '{decrypted.decode()}'")
        print(f"Decrypted length: {len(decrypted)} bytes")
        
        if decrypted == message:
            print("✓ Decrypted message matches original")
        else:
            print("✗ Decrypted message does not match original")
    except Exception as e:
        print(f"Decryption error: {e}")
        
        # Let's try to manually decrypt and handle padding
        print(f"\nAttempting manual decryption...")
        plaintext = bytearray()
        for i in range(0, len(encrypted), 64):
            block = encrypted[i:i+64]
            decrypted_block = cipher.decrypt_block(block)
            plaintext.extend(decrypted_block)
            
        print(f"Raw decrypted length: {len(plaintext)} bytes")
        print(f"Last byte (padding length?): {plaintext[-1]}")
        
        # Try to manually handle padding
        padding_value = plaintext[-1]
        if 1 <= padding_value <= 64:
            # Check a few padding bytes
            valid_padding = True
            for i in range(1, min(padding_value + 1, 5)):
                if plaintext[-i] != padding_value:
                    valid_padding = False
                    print(f"Invalid padding at position -{i}: expected {padding_value}, got {plaintext[-i]}")
                    break
            
            if valid_padding:
                unpadded = plaintext[:-padding_value]
                print(f"Manually unpadded length: {len(unpadded)} bytes")
                print(f"Manually unpadded text: '{unpadded.decode(errors='replace')}'")
        else:
            print(f"Padding value {padding_value} is outside valid range (1-64)")

if __name__ == "__main__":
    main()
