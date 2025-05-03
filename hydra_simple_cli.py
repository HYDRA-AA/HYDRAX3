#!/usr/bin/env python3
"""
Simple HYDRA Command-Line Tool

This is a simplified version of the HYDRA encryption algorithm
for demonstration purposes.
"""

import os
import sys
import hashlib
import binascii
import argparse
import getpass
from pathlib import Path

# Import our simplified HYDRA implementation
from simple_hydra import encrypt, decrypt, pad_message, unpad_message

def generate_key(size=32):
    """Generate a random key."""
    return os.urandom(size)

def derive_key_from_password(password, salt=None, iterations=100000, key_size=32):
    """Derive a key from a password using PBKDF2."""
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations=iterations,
        dklen=key_size
    )
    return key, salt

def save_key(key, filename):
    """Save a key to a file."""
    with open(filename, 'wb') as f:
        f.write(key)

def load_key(filename):
    """Load a key from a file."""
    with open(filename, 'rb') as f:
        return f.read()

def encrypt_file(input_path, output_path, key):
    """Encrypt a file."""
    with open(input_path, 'rb') as f:
        plaintext = f.read()
    
    ciphertext = encrypt(plaintext, key)
    
    with open(output_path, 'wb') as f:
        f.write(ciphertext)

def decrypt_file(input_path, output_path, key):
    """Decrypt a file."""
    with open(input_path, 'rb') as f:
        ciphertext = f.read()
    
    plaintext = decrypt(ciphertext, key)
    
    with open(output_path, 'wb') as f:
        f.write(plaintext)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Simple HYDRA Encryption Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Generate key command
    gen_parser = subparsers.add_parser('generate-key', help='Generate a random key')
    gen_parser.add_argument('--output', '-o', required=True, help='Output file for the key')
    gen_parser.add_argument('--size', type=int, choices=[256, 512], default=256, help='Key size in bits')
    
    # Password key command
    pass_parser = subparsers.add_parser('password-key', help='Generate a key from a password')
    pass_parser.add_argument('--output', '-o', required=True, help='Output file for the key')
    pass_parser.add_argument('--salt', help='Salt file (optional)')
    pass_parser.add_argument('--iterations', type=int, default=100000, help='PBKDF2 iterations')
    pass_parser.add_argument('--size', type=int, choices=[256, 512], default=256, help='Key size in bits')
    
    # Encrypt command
    enc_parser = subparsers.add_parser('encrypt', help='Encrypt a file')
    enc_parser.add_argument('--input', '-i', required=True, help='Input file')
    enc_parser.add_argument('--output', '-o', help='Output file (defaults to input.hydra)')
    enc_parser.add_argument('--key', '-k', required=True, help='Key file')
    
    # Decrypt command
    dec_parser = subparsers.add_parser('decrypt', help='Decrypt a file')
    dec_parser.add_argument('--input', '-i', required=True, help='Input file')
    dec_parser.add_argument('--output', '-o', help='Output file (defaults to removing .hydra extension)')
    dec_parser.add_argument('--key', '-k', required=True, help='Key file')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'generate-key':
        key_size = args.size // 8  # Convert from bits to bytes
        key = generate_key(key_size)
        save_key(key, args.output)
        print(f"Generated {args.size}-bit key and saved to {args.output}")
    
    elif args.command == 'password-key':
        key_size = args.size // 8  # Convert from bits to bytes
        password = getpass.getpass("Enter password: ")
        verify = getpass.getpass("Verify password: ")
        
        if password != verify:
            print("Error: Passwords do not match")
            return 1
        
        salt = None
        if args.salt:
            if os.path.exists(args.salt):
                with open(args.salt, 'rb') as f:
                    salt = f.read()
        
        key, salt = derive_key_from_password(password, salt, args.iterations, key_size)
        
        save_key(key, args.output)
        print(f"Generated {args.size}-bit key from password and saved to {args.output}")
        
        if args.salt:
            with open(args.salt, 'wb') as f:
                f.write(salt)
            print(f"Saved salt to {args.salt}")
    
    elif args.command == 'encrypt':
        key = load_key(args.key)
        output = args.output or f"{args.input}.hydra"
        
        encrypt_file(args.input, output, key)
        print(f"Encrypted {args.input} to {output}")
    
    elif args.command == 'decrypt':
        key = load_key(args.key)
        
        if not args.output:
            if args.input.endswith('.hydra'):
                output = args.input[:-6]
            else:
                output = f"{args.input}.decrypted"
        else:
            output = args.output
        
        decrypt_file(args.input, output, key)
        print(f"Decrypted {args.input} to {output}")
    
    else:
        print("Error: Please specify a command")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
