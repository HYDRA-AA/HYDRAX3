#!/usr/bin/env python3
"""
Utilities for key management in HYDRA encryption.
"""

import os
import hashlib
import base64
from typing import Tuple, Union

def generate_key(size: int = 32) -> bytes:
    """
    Generate a cryptographically secure random key.
    
    Args:
        size: Key size in bytes (32 for 256-bit, 64 for 512-bit)
    
    Returns:
        Random key as bytes
        
    Raises:
        ValueError: If key size is invalid
    """
    if size not in (32, 64):
        raise ValueError("Key size must be either 32 bytes (256 bits) or 64 bytes (512 bits)")
    
    return os.urandom(size)

def derive_key_from_password(
    password: str, 
    salt: Union[bytes, None] = None, 
    iterations: int = 100000, 
    key_size: int = 32
) -> Tuple[bytes, bytes]:
    """
    Derive a key from a password using PBKDF2.
    
    Args:
        password: Password to derive key from
        salt: Salt for key derivation (generated if None)
        iterations: Number of iterations for PBKDF2
        key_size: Key size in bytes (32 for 256-bit, 64 for 512-bit)
    
    Returns:
        Tuple of (derived key, salt)
        
    Raises:
        ValueError: If key size is invalid
    """
    if key_size not in (32, 64):
        raise ValueError("Key size must be either 32 bytes (256 bits) or 64 bytes (512 bits)")
    
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

def encode_key(key: bytes) -> str:
    """
    Encode a key as a base64 string.
    
    Args:
        key: Key to encode
    
    Returns:
        Base64-encoded key
    """
    return base64.b64encode(key).decode('ascii')

def decode_key(encoded_key: str) -> bytes:
    """
    Decode a base64-encoded key.
    
    Args:
        encoded_key: Base64-encoded key
    
    Returns:
        Decoded key as bytes
    """
    return base64.b64decode(encoded_key.encode('ascii'))

def save_key_to_file(key: bytes, filename: str) -> None:
    """
    Save a key to a file.
    
    Args:
        key: Key to save
        filename: File to save key to
    """
    with open(filename, 'wb') as f:
        f.write(key)

def load_key_from_file(filename: str) -> bytes:
    """
    Load a key from a file.
    
    Args:
        filename: File to load key from
    
    Returns:
        Key loaded from file
    """
    with open(filename, 'rb') as f:
        return f.read()

def save_key_to_file_base64(key: bytes, filename: str) -> None:
    """
    Save a key to a file as base64.
    
    Args:
        key: Key to save
        filename: File to save key to
    """
    encoded_key = encode_key(key)
    with open(filename, 'w') as f:
        f.write(encoded_key)

def load_key_from_file_base64(filename: str) -> bytes:
    """
    Load a base64-encoded key from a file.
    
    Args:
        filename: File to load key from
    
    Returns:
        Key loaded from file
    """
    with open(filename, 'r') as f:
        encoded_key = f.read().strip()
    
    return decode_key(encoded_key)
