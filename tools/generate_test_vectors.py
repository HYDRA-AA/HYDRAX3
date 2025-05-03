#!/usr/bin/env python3
"""
Generate test vectors for the HYDRA encryption algorithm.

This script creates test vectors that can be used to verify that different
implementations of the HYDRA algorithm produce the same results.
"""

import os
import sys
import json
import binascii

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import HydraCipher

def hex_to_bytes(hex_string):
    """Convert a hex string to bytes."""
    return binascii.unhexlify(hex_string)

def bytes_to_hex(data):
    """Convert bytes to a hex string."""
    return binascii.hexlify(data).decode('ascii')

def generate_ciphertext(key_hex, plaintext_hex):
    """Generate ciphertext for a given key and plaintext."""
    key = hex_to_bytes(key_hex)
    plaintext = hex_to_bytes(plaintext_hex)
    
    cipher = HydraCipher(key)
    ciphertext = cipher.encrypt(plaintext)
    
    return bytes_to_hex(ciphertext)

def update_test_vectors_file(input_file, output_file):
    """Update the test vectors file with actual ciphertext values."""
    with open(input_file, 'r') as f:
        test_vectors_data = json.load(f)
    
    # Generate the ciphertext for each test vector
    for vector in test_vectors_data['vectors']:
        key_hex = vector['key']
        plaintext_hex = vector['plaintext']
        
        # Generate the ciphertext
        ciphertext_hex = generate_ciphertext(key_hex, plaintext_hex)
        
        # Update the test vector
        vector['ciphertext'] = ciphertext_hex
    
    # Write the updated test vectors to the output file
    with open(output_file, 'w') as f:
        json.dump(test_vectors_data, f, indent=2)
    
    print(f"Test vectors with actual ciphertext values written to {output_file}")

def main():
    """Main function."""
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_root, 'tests', 'vectors', 'test_vectors.json')
    output_file = os.path.join(project_root, 'tests', 'vectors', 'hydra_test_vectors.json')
    
    # Update the test vectors file
    update_test_vectors_file(input_file, output_file)

if __name__ == '__main__':
    main()
