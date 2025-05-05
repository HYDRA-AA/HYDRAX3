"""
HYDRA Cryptography Module

A comprehensive implementation of the HYDRA encryption algorithm
featuring JIT acceleration, metadata-enhanced reliable decryption,
and multi-threading for large datasets.

This package provides:
- Core encryption/decryption functions
- File encryption utilities
- Utilities for key generation and management

Example usage:
    from hydra_crypto import encrypt, decrypt
    
    key = os.urandom(32)  # Generate a 256-bit key
    encrypted = encrypt(b"Secret message", key)
    original = decrypt(encrypted, key)
"""

import os
from typing import Optional, Union, BinaryIO

# Import core functions from the unified implementation
from .core import (
    encrypt as _encrypt,
    decrypt as _decrypt,
    UnifiedHydraCipher,
    measure_complexity,
    calculate_additional_rounds,
    pad_data,
    unpad_data,
    derive_round_key,
)

# Re-export public API
__all__ = [
    'encrypt', 'decrypt', 'encrypt_file', 'decrypt_file',
    'UnifiedHydraCipher', 'generate_key', 'derive_key_from_password'
]

# Version information
__version__ = '1.0.0'
__author__ = 'HYDRA Cryptography Team'

def encrypt(data: bytes, key: bytes, parallel: bool = True) -> bytes:
    """
    Encrypt data using the HYDRA algorithm.
    
    Args:
        data: Data to encrypt
        key: Encryption key (32 bytes recommended)
        parallel: Whether to use parallel processing for large data
        
    Returns:
        Encrypted data with metadata for reliable decryption
    """
    return _encrypt(data, key, parallel)

def decrypt(data: bytes, key: bytes, parallel: bool = True) -> bytes:
    """
    Decrypt data using the HYDRA algorithm.
    
    Args:
        data: Data to decrypt
        key: Decryption key (same as encryption key)
        parallel: Whether to use parallel processing for large data
        
    Returns:
        Original decrypted data
    """
    return _decrypt(data, key, parallel)

def generate_key(size: int = 32) -> bytes:
    """
    Generate a cryptographically secure random key.
    
    Args:
        size: Key size in bytes (32 for 256-bit, 64 for 512-bit)
        
    Returns:
        Random key of specified size
    """
    return os.urandom(size)

def derive_key_from_password(password: str, salt: Optional[bytes] = None, 
                             iterations: int = 100000) -> tuple[bytes, bytes]:
    """
    Derive a key from a password using PBKDF2.
    
    Args:
        password: User password
        salt: Salt bytes (generated randomly if None)
        iterations: Number of PBKDF2 iterations
        
    Returns:
        Tuple of (derived_key, salt)
    """
    import hashlib
    
    if salt is None:
        salt = os.urandom(16)
        
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
    return key, salt

def encrypt_file(input_path: Union[str, BinaryIO], output_path: Union[str, BinaryIO], 
                 key: bytes, parallel: bool = True) -> None:
    """
    Encrypt a file with HYDRA.
    
    Args:
        input_path: Path to input file or file-like object
        output_path: Path to output file or file-like object
        key: Encryption key
        parallel: Whether to use parallel processing
    """
    # Handle string paths or file objects
    if isinstance(input_path, str):
        with open(input_path, 'rb') as infile:
            data = infile.read()
    else:
        data = input_path.read()
    
    # Encrypt the data
    encrypted_data = encrypt(data, key, parallel)
    
    # Write to output
    if isinstance(output_path, str):
        with open(output_path, 'wb') as outfile:
            outfile.write(encrypted_data)
    else:
        output_path.write(encrypted_data)

def decrypt_file(input_path: Union[str, BinaryIO], output_path: Union[str, BinaryIO], 
                 key: bytes, parallel: bool = True) -> None:
    """
    Decrypt a file with HYDRA.
    
    Args:
        input_path: Path to encrypted file or file-like object
        output_path: Path to output file or file-like object
        key: Decryption key
        parallel: Whether to use parallel processing
    """
    # Handle string paths or file objects
    if isinstance(input_path, str):
        with open(input_path, 'rb') as infile:
            data = infile.read()
    else:
        data = input_path.read()
    
    # Decrypt the data
    decrypted_data = decrypt(data, key, parallel)
    
    # Write to output
    if isinstance(output_path, str):
        with open(output_path, 'wb') as outfile:
            outfile.write(decrypted_data)
    else:
        output_path.write(decrypted_data)
