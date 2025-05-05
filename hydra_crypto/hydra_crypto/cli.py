#!/usr/bin/env python3
"""
HYDRA File Encryptor/Decryptor

A command-line tool for encrypting and decrypting files using the 
enhanced HYDRA encryption algorithm with JIT acceleration and
reliable metadata-based decryption.
"""

import os
import sys
import time
import argparse
import getpass
import hashlib
from typing import Tuple
import base64

# Import from the hydra_crypto package
from hydra_crypto import encrypt, decrypt, derive_key_from_password

# The key derivation functionality is now imported from the package

def encrypt_file(input_file: str, output_file: str, password: str) -> None:
    """
    Encrypt a file using HYDRA encryption.
    
    Args:
        input_file: Path to the file to encrypt
        output_file: Path where encrypted file will be saved
        password: Encryption password
    """
    try:
        # Read the input file
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # Generate a key from the password
        key, salt = derive_key_from_password(password)
        
        # Record start time for performance measurement
        start_time = time.time()
        
        # Encrypt the data
        encrypted_data = encrypt(data, key)
        
        # Calculate encryption time
        encryption_time = time.time() - start_time
        
        # Format the output with salt and encrypted data
        output_data = b"HYDRA1:"
        output_data += base64.b64encode(salt) + b":"
        output_data += base64.b64encode(encrypted_data)
        
        # Write the encrypted data to the output file
        with open(output_file, 'wb') as f:
            f.write(output_data)
        
        # Print summary
        print(f"Encryption completed successfully:")
        print(f"- Original size: {len(data):,} bytes")
        print(f"- Encrypted size: {len(encrypted_data):,} bytes")
        print(f"- Encryption time: {encryption_time:.2f} seconds")
        print(f"- Throughput: {(len(data) / 1024 / 1024) / max(0.001, encryption_time):.2f} MB/s")
        print(f"- Output saved to: {output_file}")
    
    except Exception as e:
        print(f"Error during encryption: {e}")
        sys.exit(1)

def decrypt_file(input_file: str, output_file: str, password: str) -> None:
    """
    Decrypt a file using HYDRA encryption.
    
    Args:
        input_file: Path to the encrypted file
        output_file: Path where decrypted file will be saved
        password: Decryption password
    """
    try:
        # Read the encrypted file
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # Parse the header format: "HYDRA1:<base64_salt>:<base64_encrypted_data>"
        parts = data.split(b":", 2)
        
        if len(parts) != 3 or parts[0] != b"HYDRA1":
            print("Error: Not a valid HYDRA encrypted file")
            sys.exit(1)
        
        # Extract salt and encrypted data
        salt = base64.b64decode(parts[1])
        encrypted_data = base64.b64decode(parts[2])
        
        # Derive key from password and salt
        key, _ = derive_key_from_password(password, salt)
        
        # Record start time for performance measurement
        start_time = time.time()
        
        # Decrypt the data
        decrypted_data = decrypt(encrypted_data, key)
        
        # Calculate decryption time
        decryption_time = time.time() - start_time
        
        # Write the decrypted data to the output file
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
        
        # Print summary
        print(f"Decryption completed successfully:")
        print(f"- Encrypted size: {len(encrypted_data):,} bytes")
        print(f"- Decrypted size: {len(decrypted_data):,} bytes")
        print(f"- Decryption time: {decryption_time:.2f} seconds")
        print(f"- Throughput: {(len(encrypted_data) / 1024 / 1024) / max(0.001, decryption_time):.2f} MB/s")
        print(f"- Output saved to: {output_file}")
    
    except Exception as e:
        print(f"Error during decryption: {e}")
        sys.exit(1)

def main():
    """Main function handling command-line arguments."""
    parser = argparse.ArgumentParser(description="HYDRA File Encryptor/Decryptor")
    
    # Create mutually exclusive group for encrypt/decrypt
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', '--encrypt', action='store_true', help='Encrypt a file')
    group.add_argument('-d', '--decrypt', action='store_true', help='Decrypt a file')
    
    # Input and output files
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('-p', '--password', help='Password (not recommended, will prompt if omitted)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get password
    password = args.password
    if not password:
        password = getpass.getpass(prompt='Enter password: ')
    
    # Encrypt or decrypt
    if args.encrypt:
        encrypt_file(args.input, args.output, password)
    else:
        decrypt_file(args.input, args.output, password)

if __name__ == "__main__":
    main()
