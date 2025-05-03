#!/usr/bin/env python3
"""
Utilities for file operations in HYDRA encryption.
"""

import os
import struct
from typing import Callable, BinaryIO

def encrypt_file(
    input_path: str,
    output_path: str,
    encrypt_func: Callable[[bytes], bytes],
    chunk_size: int = 1024 * 1024  # 1 MB
) -> None:
    """
    Encrypt a file using the provided encryption function.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        encrypt_func: Function that encrypts a chunk of data
        chunk_size: Size of chunks to process at once
    """
    # Get file size
    file_size = os.path.getsize(input_path)
    
    with open(input_path, 'rb') as in_file, open(output_path, 'wb') as out_file:
        # Write file size header (for decryption padding validation)
        out_file.write(struct.pack('<Q', file_size))
        
        # Process file in chunks
        while True:
            chunk = in_file.read(chunk_size)
            if not chunk:
                break
            
            # Encrypt chunk
            encrypted_chunk = encrypt_func(chunk)
            out_file.write(encrypted_chunk)

def decrypt_file(
    input_path: str,
    output_path: str,
    decrypt_func: Callable[[bytes], bytes],
    chunk_size: int = 1024 * 1024  # 1 MB
) -> None:
    """
    Decrypt a file using the provided decryption function.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        decrypt_func: Function that decrypts a chunk of data
        chunk_size: Size of chunks to process at once
    """
    with open(input_path, 'rb') as in_file, open(output_path, 'wb') as out_file:
        # Read file size header
        file_size_bytes = in_file.read(8)
        original_file_size = struct.unpack('<Q', file_size_bytes)[0]
        
        # Process file in chunks
        bytes_written = 0
        
        while True:
            chunk = in_file.read(chunk_size)
            if not chunk:
                break
            
            # Decrypt chunk
            decrypted_chunk = decrypt_func(chunk)
            
            # For the last chunk, truncate to the original file size
            if bytes_written + len(decrypted_chunk) > original_file_size:
                decrypted_chunk = decrypted_chunk[:original_file_size - bytes_written]
            
            out_file.write(decrypted_chunk)
            bytes_written += len(decrypted_chunk)
            
            # Stop if we've written the entire file
            if bytes_written >= original_file_size:
                break

def process_file_in_place(
    file_path: str,
    process_func: Callable[[bytes], bytes],
    chunk_size: int = 1024 * 1024  # 1 MB
) -> None:
    """
    Process a file in place using the provided function.
    
    Args:
        file_path: Path to file
        process_func: Function that processes a chunk of data
        chunk_size: Size of chunks to process at once
    """
    # Create a temporary output file
    temp_path = file_path + '.temp'
    
    try:
        # Process the file to the temporary file
        with open(file_path, 'rb') as in_file, open(temp_path, 'wb') as out_file:
            while True:
                chunk = in_file.read(chunk_size)
                if not chunk:
                    break
                
                # Process chunk
                processed_chunk = process_func(chunk)
                out_file.write(processed_chunk)
        
        # Replace the original file with the processed file
        os.replace(temp_path, file_path)
    except Exception:
        # Clean up temporary file if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def is_encrypted_file(file_path: str) -> bool:
    """
    Check if a file appears to be encrypted with HYDRA.
    
    This is a simple heuristic and may not be 100% accurate.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file appears to be encrypted, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            # Try to read the file size header
            file_size_bytes = f.read(8)
            
            if len(file_size_bytes) != 8:
                return False
            
            # Check if the remaining file size is a multiple of the block size (64 bytes)
            file_size = os.path.getsize(file_path) - 8
            
            if file_size % 64 != 0 or file_size == 0:
                return False
            
            return True
    except Exception:
        return False

def get_file_info(file_path: str) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path: Path to file
    
    Returns:
        Dictionary with file information
    """
    return {
        'path': file_path,
        'size': os.path.getsize(file_path),
        'encrypted': is_encrypted_file(file_path),
        'exists': os.path.exists(file_path),
        'is_file': os.path.isfile(file_path),
        'is_dir': os.path.isdir(file_path),
    }
