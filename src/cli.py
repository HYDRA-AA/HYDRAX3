#!/usr/bin/env python3
"""
Command-line interface for HYDRA encryption.
"""

import os
import sys
import argparse
import getpass
from typing import List, Optional

from src.core import HydraCipher
from src.utils import key_utils, file_utils

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments (uses sys.argv if None)
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="HYDRA Encryption Tool",
        epilog="NOTE: This is experimental encryption software and should not be used for sensitive data."
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Key generation command
    key_gen_parser = subparsers.add_parser("generate-key", help="Generate an encryption key")
    key_gen_parser.add_argument(
        "--size", type=int, choices=[256, 512], default=256,
        help="Key size in bits (256 or 512)"
    )
    key_gen_parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output file for the key"
    )
    key_gen_parser.add_argument(
        "--base64", action="store_true",
        help="Save key in base64 format"
    )
    
    # Password-based key generation command
    pass_key_parser = subparsers.add_parser("password-key", help="Generate a key from a password")
    pass_key_parser.add_argument(
        "--size", type=int, choices=[256, 512], default=256,
        help="Key size in bits (256 or 512)"
    )
    pass_key_parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output file for the key"
    )
    pass_key_parser.add_argument(
        "--salt-file", type=str,
        help="File to save/load salt"
    )
    pass_key_parser.add_argument(
        "--iterations", type=int, default=100000,
        help="Number of iterations for PBKDF2"
    )
    pass_key_parser.add_argument(
        "--base64", action="store_true",
        help="Save key in base64 format"
    )
    
    # Encryption command
    encrypt_parser = subparsers.add_parser("encrypt", help="Encrypt a file")
    encrypt_parser.add_argument(
        "--key", "-k", type=str, required=True,
        help="Key file"
    )
    encrypt_parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input file to encrypt"
    )
    encrypt_parser.add_argument(
        "--output", "-o", type=str,
        help="Output file (defaults to input file with .hydra extension)"
    )
    encrypt_parser.add_argument(
        "--base64", action="store_true",
        help="Key file is in base64 format"
    )
    
    # Decryption command
    decrypt_parser = subparsers.add_parser("decrypt", help="Decrypt a file")
    decrypt_parser.add_argument(
        "--key", "-k", type=str, required=True,
        help="Key file"
    )
    decrypt_parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input file to decrypt"
    )
    decrypt_parser.add_argument(
        "--output", "-o", type=str,
        help="Output file (defaults to input file without .hydra extension)"
    )
    decrypt_parser.add_argument(
        "--base64", action="store_true",
        help="Key file is in base64 format"
    )
    
    # File info command
    info_parser = subparsers.add_parser("info", help="Get information about a file")
    info_parser.add_argument(
        "file", type=str,
        help="File to get information about"
    )
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    return parser.parse_args(args)

def generate_key_cmd(args: argparse.Namespace) -> int:
    """
    Generate a random key.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        # Convert key size from bits to bytes
        key_size = args.size // 8
        
        # Generate the key
        key = key_utils.generate_key(key_size)
        
        # Save the key
        if args.base64:
            key_utils.save_key_to_file_base64(key, args.output)
        else:
            key_utils.save_key_to_file(key, args.output)
        
        print(f"Generated {args.size}-bit key and saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error generating key: {e}", file=sys.stderr)
        return 1

def password_key_cmd(args: argparse.Namespace) -> int:
    """
    Generate a key from a password.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        # Convert key size from bits to bytes
        key_size = args.size // 8
        
        # Get the password
        password = getpass.getpass("Enter password: ")
        password_verify = getpass.getpass("Verify password: ")
        
        if password != password_verify:
            print("Passwords do not match", file=sys.stderr)
            return 1
        
        # Load salt if provided
        salt = None
        if args.salt_file and os.path.exists(args.salt_file):
            with open(args.salt_file, 'rb') as f:
                salt = f.read()
        
        # Derive the key
        key, salt = key_utils.derive_key_from_password(
            password, salt, args.iterations, key_size
        )
        
        # Save the salt if provided
        if args.salt_file:
            with open(args.salt_file, 'wb') as f:
                f.write(salt)
            print(f"Saved salt to {args.salt_file}")
        
        # Save the key
        if args.base64:
            key_utils.save_key_to_file_base64(key, args.output)
        else:
            key_utils.save_key_to_file(key, args.output)
        
        print(f"Generated {args.size}-bit key from password and saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error generating key: {e}", file=sys.stderr)
        return 1

def encrypt_cmd(args: argparse.Namespace) -> int:
    """
    Encrypt a file.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        # Determine output file
        output_file = args.output
        if not output_file:
            output_file = args.input + '.hydra'
        
        # Load the key
        if args.base64:
            key = key_utils.load_key_from_file_base64(args.key)
        else:
            key = key_utils.load_key_from_file(args.key)
        
        # Create cipher
        cipher = HydraCipher(key)
        
        # Encrypt the file
        print(f"Encrypting {args.input} to {output_file}...")
        file_utils.encrypt_file(
            args.input,
            output_file,
            cipher.encrypt
        )
        
        print(f"Encrypted {args.input} to {output_file}")
        return 0
    except Exception as e:
        print(f"Error encrypting file: {e}", file=sys.stderr)
        return 1

def decrypt_cmd(args: argparse.Namespace) -> int:
    """
    Decrypt a file.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        # Determine output file
        output_file = args.output
        if not output_file:
            if args.input.endswith('.hydra'):
                output_file = args.input[:-6]
            else:
                output_file = args.input + '.decrypted'
        
        # Load the key
        if args.base64:
            key = key_utils.load_key_from_file_base64(args.key)
        else:
            key = key_utils.load_key_from_file(args.key)
        
        # Create cipher
        cipher = HydraCipher(key)
        
        # Decrypt the file
        print(f"Decrypting {args.input} to {output_file}...")
        file_utils.decrypt_file(
            args.input,
            output_file,
            cipher.decrypt
        )
        
        print(f"Decrypted {args.input} to {output_file}")
        return 0
    except Exception as e:
        print(f"Error decrypting file: {e}", file=sys.stderr)
        return 1

def info_cmd(args: argparse.Namespace) -> int:
    """
    Get information about a file.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        # Get file information
        info = file_utils.get_file_info(args.file)
        
        # Print information
        print(f"File: {info['path']}")
        print(f"Size: {info['size']} bytes")
        print(f"Exists: {info['exists']}")
        print(f"Is file: {info['is_file']}")
        print(f"Is directory: {info['is_dir']}")
        print(f"Appears to be HYDRA encrypted: {info['encrypted']}")
        
        return 0
    except Exception as e:
        print(f"Error getting file information: {e}", file=sys.stderr)
        return 1

def version_cmd(args: argparse.Namespace) -> int:
    """
    Show version information.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    from src import __version__
    print(f"HYDRA Encryption v{__version__}")
    print("Copyright (c) 2025 HYDRA Encryption Project Contributors")
    print("This is experimental encryption software and should not be used for sensitive data.")
    return 0

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point.
    
    Args:
        args: Command-line arguments (uses sys.argv if None)
    
    Returns:
        Exit code
    """
    parsed_args = parse_args(args)
    
    # Dispatch to appropriate command
    if parsed_args.command == "generate-key":
        return generate_key_cmd(parsed_args)
    elif parsed_args.command == "password-key":
        return password_key_cmd(parsed_args)
    elif parsed_args.command == "encrypt":
        return encrypt_cmd(parsed_args)
    elif parsed_args.command == "decrypt":
        return decrypt_cmd(parsed_args)
    elif parsed_args.command == "info":
        return info_cmd(parsed_args)
    elif parsed_args.command == "version":
        return version_cmd(parsed_args)
    else:
        print("Please specify a command. Use --help for more information.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
