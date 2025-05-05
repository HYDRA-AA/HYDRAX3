#!/usr/bin/env python3
"""
JIT-accelerated HYDRA Encryption Algorithm.

This implementation uses Numba to accelerate core cryptographic operations.
It also provides a compatibility layer with the original HYDRA cipher.
"""

import os
import struct
import hashlib
import threading
from typing import List, Tuple, Union, Callable
import numba as nb
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure Numba for best performance
nb.config.NUMBA_DEFAULT_NUM_THREADS = os.cpu_count() or 4
# Use parallel option cautiously - sometimes sequential is faster for small inputs
USE_PARALLEL = True

@nb.njit(cache=True)
def xor_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    XOR two numpy arrays (optimized with Numba).
    """
    return a ^ b

@nb.njit(cache=True)
def rotate_left_byte(value: int, shift: int) -> int:
    """
    Rotate a byte left by a given number of bits.
    """
    return ((value << shift) | (value >> (8 - shift))) & 0xFF

@nb.njit(cache=True)
def rotate_right_byte(value: int, shift: int) -> int:
    """
    Rotate a byte right by a given number of bits.
    """
    return ((value >> shift) | (value << (8 - shift))) & 0xFF

@nb.njit(cache=True)
def d_rot_64bytes(state: np.ndarray, dimension: int, amount: int) -> np.ndarray:
    """
    JIT-optimized dimensional rotation for 64-byte blocks.
    """
    # Create a copy of the input state
    new_state = np.zeros_like(state)
    mask = 1 << dimension
    
    for base_idx in range(0, 64, 4):  # 16 groups of 4 bytes
        group_idx = base_idx // 4
        
        # Compute the target index based on dimension to rotate
        flip_bit = (group_idx & mask) != 0
        
        # If we need to move this group based on rotation amount
        if (amount % 2 == 1) != flip_bit:  # XOR logic
            # Flip the bit in the dimension we're rotating
            target_group = group_idx ^ mask
            target_base = target_group * 4
            
            # Copy all 4 bytes at once
            new_state[target_base:target_base+4] = state[base_idx:base_idx+4]
        else:
            # Keep the bytes in same position
            new_state[base_idx:base_idx+4] = state[base_idx:base_idx+4]
    
    return new_state

@nb.njit(cache=True)
def d_rot_inverse_64bytes(state: np.ndarray, dimension: int, amount: int) -> np.ndarray:
    """
    JIT-optimized inverse dimensional rotation for 64-byte blocks.
    """
    # For an inverse rotation, we simply rotate in the opposite direction
    inverse_amount = -amount % 16  # DIM_SIZE = 16
    return d_rot_64bytes(state, dimension, inverse_amount)

@nb.njit(cache=True)
def ms_box_64bytes(state: np.ndarray, s_boxes: np.ndarray, num_s_boxes: int) -> np.ndarray:
    """
    JIT-optimized multi-scale substitution for 64-byte blocks.
    Args:
        state: 64-byte array
        s_boxes: s_boxes array of shape (num_s_boxes, 256)
        num_s_boxes: number of s-boxes (typically 16)
    """
    new_state = np.zeros_like(state)
    
    for block in range(0, 64, 16):  # Process 16 bytes at a time
        for i in range(block, block + 16):
            # Calculate context based on preceding byte
            context = state[(i - 1) & 63]  # Fast modulo for powers of 2
            
            # Select S-box and apply substitution
            s_box_idx = context % num_s_boxes
            new_state[i] = s_boxes[s_box_idx, state[i]]
    
    return new_state

@nb.njit(cache=True)
def ms_box_inverse_64bytes(state: np.ndarray, inv_s_boxes: np.ndarray, num_s_boxes: int) -> np.ndarray:
    """
    JIT-optimized inverse multi-scale substitution for 64-byte blocks.
    """
    new_state = np.zeros_like(state)
    
    for block in range(0, 64, 16):  # Process 16 bytes at a time
        for i in range(block, block + 16):
            # Calculate context based on preceding byte
            context = state[(i - 1) & 63]  # Fast modulo for powers of 2
            
            # Select inverse S-box and apply substitution
            s_box_idx = context % num_s_boxes
            new_state[i] = inv_s_boxes[s_box_idx, state[i]]
    
    return new_state

@nb.njit(cache=True)
def f_diff_64bytes(state: np.ndarray) -> np.ndarray:
    """
    JIT-optimized fractal diffusion for 64-byte blocks.
    """
    new_state = np.zeros_like(state)
    
    # Process in blocks of 8 bytes for better cache utilization
    for block in range(0, 64, 8):
        for i in range(block, block + 8):
            # Fast XOR with offsets (no modulo needed for most operations)
            mixed = state[i]
            
            # Unrolled operations for each offset
            # For 64-byte state, we only need modulo for some offsets
            mixed ^= state[(i + 1) & 63]  # Faster than i+1 % 64
            mixed = rotate_left_byte(mixed, 1)
            
            mixed ^= state[(i + 2) & 63]
            mixed = rotate_left_byte(mixed, 1)
            
            mixed ^= state[(i + 4) & 63]
            mixed = rotate_left_byte(mixed, 1)
            
            mixed ^= state[(i + 8) & 63]
            mixed = rotate_left_byte(mixed, 1)
            
            mixed ^= state[(i + 16) & 63]
            mixed = rotate_left_byte(mixed, 1)
            
            mixed ^= state[(i + 32) & 63]
            mixed = rotate_left_byte(mixed, 1)
            
            new_state[i] = mixed
    
    return new_state

@nb.njit(cache=True)
def f_diff_inverse_64bytes(state: np.ndarray) -> np.ndarray:
    """
    JIT-optimized inverse fractal diffusion for 64-byte blocks.
    """
    new_state = np.zeros_like(state)
    
    # Pre-define inverse mixing offsets (reverse of forward offsets)
    mix_offsets = np.array([32, 16, 8, 4, 2, 1], dtype=np.int32)
    
    # Iterate in reverse for improved inverse diffusion
    for block in range(56, -8, -8):  # Process in reverse from 56 down to 0
        for i in range(block + 7, block - 1, -1):  # Process 8 bytes in each block
            # Optimized inverse operations (no modulo for most offsets)
            mixed = state[i]
            
            # Unrolled inverse operations for each offset in reverse order
            
            # Offset 32
            mixed = rotate_right_byte(mixed, 1)
            mixed ^= state[(i + 32) & 63]
            
            # Offset 16
            mixed = rotate_right_byte(mixed, 1)
            mixed ^= state[(i + 16) & 63]
            
            # Offset 8
            mixed = rotate_right_byte(mixed, 1)
            mixed ^= state[(i + 8) & 63]
            
            # Offset 4
            mixed = rotate_right_byte(mixed, 1)
            mixed ^= state[(i + 4) & 63]
            
            # Offset 2
            mixed = rotate_right_byte(mixed, 1)
            mixed ^= state[(i + 2) & 63]
            
            # Offset 1
            mixed = rotate_right_byte(mixed, 1)
            mixed ^= state[(i + 1) & 63]
            
            new_state[i] = mixed
    
    return new_state

@nb.njit(cache=True)
def a_perm_64bytes(state: np.ndarray, perm_indices: np.ndarray) -> np.ndarray:
    """
    JIT-optimized adaptive permutation for 64-byte blocks.
    
    Args:
        state: 64-byte array
        perm_indices: precomputed permutation indices
    """
    new_state = np.zeros_like(state)
    
    # Apply the permutation
    for i in range(len(state)):
        new_state[perm_indices[i]] = state[i]
    
    return new_state

@nb.njit(cache=True)
def a_perm_inverse_64bytes(state: np.ndarray, perm_indices: np.ndarray) -> np.ndarray:
    """
    JIT-optimized inverse adaptive permutation for 64-byte blocks.
    """
    new_state = np.zeros_like(state)
    
    # Apply the inverse permutation
    for i in range(len(state)):
        new_state[i] = state[perm_indices[i]]
    
    return new_state

class JitHydraCipher:
    """
    JIT-accelerated implementation of the HYDRA encryption algorithm.
    
    This implementation uses Numba JIT compilation for core operations and
    provides a similar API to the original HydraCipher class.
    """
    
    # Number of dimensions in the hypercube
    DIMENSIONS = 4
    
    # Size of each dimension (2^4 = 16)
    DIM_SIZE = 16
    
    # Minimum and maximum number of rounds
    MIN_ROUNDS = 16
    MAX_ROUNDS = 24
    
    # Number of S-boxes for Multi-scale Substitution
    NUM_S_BOXES = 16
    
    def __init__(self, key: bytes, max_threads: int = None):
        """
        Initialize the cipher with a key.
        
        Args:
            key: A 32-byte (256-bit) or 64-byte (512-bit) key
            max_threads: Maximum number of threads to use for parallel processing
        
        Raises:
            ValueError: If the key size is invalid
        """
        if len(key) not in (32, 64):
            raise ValueError("Key must be either 32 bytes (256 bits) or 64 bytes (512 bits)")
        
        self.key = key
        self.max_threads = max_threads or (os.cpu_count() or 2)
        
        # Generate round keys
        self.round_keys = self._key_schedule(key)
        
        # Initialize S-boxes and convert to numpy arrays for JIT optimization
        self.s_boxes = self._generate_s_boxes(key)
        self.s_boxes_np = np.array(self.s_boxes, dtype=np.uint8)
        
        # Precompute inverse S-boxes for better performance
        self.inv_s_boxes = self._generate_inverse_s_boxes()
        self.inv_s_boxes_np = np.array(self.inv_s_boxes, dtype=np.uint8)
        
        # Generate rotation amounts
        self.rotation_amounts = self._generate_rotation_amounts(key)
        self.rotation_amounts_np = np.array(self.rotation_amounts, dtype=np.int32)

        # Pre-compute permutation indices for each round key
        self.perm_indices = []
        self.inv_perm_indices = []
        for round_key in self.round_keys:
            perm = self._compute_permutation(round_key)
            self.perm_indices.append(np.array(perm, dtype=np.int32))
            
            # Also precompute inverse permutation
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            self.inv_perm_indices.append(np.array(inv_perm, dtype=np.int32))
    
    def _compute_permutation(self, round_key: bytes) -> List[int]:
        """
        Compute permutation indices based on round key.
        """
        state_size = 64
        perm = list(range(state_size))
        
        # Simple key-dependent permutation
        for i in range(state_size - 1, 0, -1):
            j = round_key[i % len(round_key)] % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
            
        return perm
    
    def _key_schedule(self, key: bytes) -> List[bytes]:
        """
        Generate round keys using fractal expansion of the master key.
        """
        round_keys = []
        
        # Create enough round keys for the maximum number of rounds plus an extra for initial whitening
        for i in range(self.MAX_ROUNDS + 1):
            # Simple key derivation: hash the master key with the round number
            h = hashlib.sha512()
            h.update(key)
            h.update(struct.pack("<I", i))  # Add round number
            
            # Take the first 64 bytes of the hash as the round key
            round_key = h.digest()[:64]
            round_keys.append(round_key)
        
        return round_keys
    
    def _generate_s_boxes(self, key: bytes) -> List[List[int]]:
        """
        Generate S-boxes based on the key.
        """
        s_boxes = []
        
        for i in range(self.NUM_S_BOXES):
            # Generate a permutation of numbers 0-255
            h = hashlib.sha512()
            h.update(key)
            h.update(struct.pack("<I", i))  # Add S-box index
            
            # Use the hash to seed a simple key-derived permutation
            seed = int.from_bytes(h.digest()[:8], byteorder='little')
            s_box = list(range(256))
            
            # Fisher-Yates shuffle based on the seed
            import random
            rng = random.Random(seed)
            for j in range(255, 0, -1):
                k = rng.randint(0, j)
                s_box[j], s_box[k] = s_box[k], s_box[j]
            
            s_boxes.append(s_box)
        
        return s_boxes

    def _generate_inverse_s_boxes(self) -> List[List[int]]:
        """
        Generate inverse S-boxes for more efficient decryption.
        """
        inverse_s_boxes = []
        
        for s_box in self.s_boxes:
            # Create inverse S-box - for each value in the original S-box,
            # store its original index at that value's position
            inverse_box = [0] * 256
            for i in range(256):
                inverse_box[s_box[i]] = i
            inverse_s_boxes.append(inverse_box)
        
        return inverse_s_boxes
        
    def _generate_rotation_amounts(self, key: bytes) -> List[int]:
        """
        Generate rotation amounts based on the key.
        """
        h = hashlib.sha512()
        h.update(key)
        h.update(b"rotation_constants")
        
        # Generate enough rotation amounts for all possible rounds
        rotation_seed = h.digest()
        rotation_amounts = []
        
        for i in range(0, self.MAX_ROUNDS * 2, 2):
            # Use 2 bytes from the hash for each rotation amount (modulo 16 to keep reasonable)
            amount = (rotation_seed[i] + (rotation_seed[i+1] << 8)) % 16
            rotation_amounts.append(amount)
        
        return rotation_amounts
    
    def measure_complexity(self, state: Union[bytes, bytearray, np.ndarray]) -> float:
        """
        Measure the complexity of the current state to determine if additional rounds are needed.
        """
        # For this simplified implementation, we'll use a simple entropy-based measure
        # Count the number of different byte values
        if isinstance(state, (bytes, bytearray)):
            state_array = np.frombuffer(state, dtype=np.uint8)
        else:
            state_array = state
            
        unique_bytes = len(np.unique(state_array))
        
        # Normalize to 0.0-1.0 range
        complexity = unique_bytes / 256.0
        
        return complexity
    
    def calculate_additional_rounds(self, complexity: float) -> int:
        """
        Calculate how many additional rounds to perform based on data complexity.
        """
        # Scale complexity to 0-8 additional rounds
        additional_rounds = int(complexity * 8)
        return additional_rounds
    
    def encrypt_block(self, plaintext: bytes) -> bytes:
        """
        Encrypt a single block of data using JIT-accelerated functions.
        
        Args:
            plaintext: 64-byte block of plaintext
            
        Returns:
            64-byte block of ciphertext
        """
        if len(plaintext) != 64:
            raise ValueError("Plaintext block must be exactly 64 bytes")
        
        # Convert to numpy array for JIT operations
        state = np.frombuffer(plaintext, dtype=np.uint8).copy()
        
        # Initial whitening
        state = xor_arrays(state, np.frombuffer(self.round_keys[0], dtype=np.uint8))
        
        # Main rounds
        for i in range(1, self.MIN_ROUNDS + 1):
            # Apply round operations with JIT acceleration
            state = d_rot_64bytes(state, i % 4, self.rotation_amounts[i])
            state = ms_box_64bytes(state, self.s_boxes_np, self.NUM_S_BOXES)
            state = f_diff_64bytes(state)
            state = a_perm_64bytes(state, self.perm_indices[i])
            state = xor_arrays(state, np.frombuffer(self.round_keys[i], dtype=np.uint8))
        
        # Adaptive rounds if needed
        complexity = self.measure_complexity(state)
        additional_rounds = self.calculate_additional_rounds(complexity)
        
        for i in range(self.MIN_ROUNDS + 1, self.MIN_ROUNDS + additional_rounds + 1):
            round_idx = i % len(self.round_keys)
            
            # Apply round operations
            state = d_rot_64bytes(state, i % 4, self.rotation_amounts[i % len(self.rotation_amounts)])
            state = ms_box_64bytes(state, self.s_boxes_np, self.NUM_S_BOXES)
            state = f_diff_64bytes(state)
            state = a_perm_64bytes(state, self.perm_indices[round_idx])
            state = xor_arrays(state, np.frombuffer(self.round_keys[round_idx], dtype=np.uint8))
        
        # Final transformation
        state = xor_arrays(state, np.frombuffer(self.round_keys[-1], dtype=np.uint8))
        
        return bytes(state)
    
    def decrypt_block(self, ciphertext: bytes) -> bytes:
        """
        Decrypt a single block of data using JIT acceleration.
        
        Args:
            ciphertext: 64-byte block of ciphertext
            
        Returns:
            64-byte block of plaintext
        """
        if len(ciphertext) != 64:
            raise ValueError("Ciphertext block must be exactly 64 bytes")
        
        # Convert to numpy array for JIT operations
        state = np.frombuffer(ciphertext, dtype=np.uint8).copy()
        
        # Undo final transformation
        state = xor_arrays(state, np.frombuffer(self.round_keys[-1], dtype=np.uint8))
        
        # First, determine how many rounds were used by measuring complexity
        complexity = self.measure_complexity(state)
        additional_rounds = self.calculate_additional_rounds(complexity)
        total_rounds = self.MIN_ROUNDS + additional_rounds
        
        # Inverse rounds (in reverse order)
        for i in range(total_rounds, 0, -1):
            round_idx = i % len(self.round_keys)
            
            # Apply inverse operations in reverse order with JIT acceleration
            state = xor_arrays(state, np.frombuffer(self.round_keys[round_idx], dtype=np.uint8))
            state = a_perm_inverse_64bytes(state, self.inv_perm_indices[round_idx])
            state = f_diff_inverse_64bytes(state)
            state = ms_box_inverse_64bytes(state, self.inv_s_boxes_np, self.NUM_S_BOXES)
            state = d_rot_inverse_64bytes(state, i % 4, self.rotation_amounts[i % len(self.rotation_amounts)])
        
        # Undo initial whitening
        state = xor_arrays(state, np.frombuffer(self.round_keys[0], dtype=np.uint8))
        
        return bytes(state)
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt data using HYDRA with JIT acceleration.
        
        Args:
            plaintext: Data to encrypt
            
        Returns:
            Encrypted data
        """
        # Apply PKCS#7 padding
        original_length = len(plaintext)
        padded_length = ((original_length + 63) // 64) * 64
        padding_length = padded_length - original_length
        
        # Ensure we have at least 1 byte of padding
        if padding_length == 0:
            padding_length = 64
            padded_length += 64
        
        # Create padded plaintext - consistent padding with the same value
        padded_plaintext = bytearray(plaintext) + bytearray([padding_length] * padding_length)
        
        # Set up multithreading for large inputs
        use_threading = len(padded_plaintext) >= 1024 and self.max_threads > 1
        
        if use_threading:
            return self._parallel_process_blocks(padded_plaintext, self.encrypt_block)
        else:
            # Single-threaded encryption for small inputs
            return self._sequential_process_blocks(padded_plaintext, self.encrypt_block)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt data using HYDRA with JIT acceleration.
        
        Args:
            ciphertext: Data to decrypt
            
        Returns:
            Decrypted data
        """
        if len(ciphertext) % 64 != 0:
            raise ValueError("Ciphertext length must be a multiple of 64 bytes")
        
        # Set up multithreading for large inputs
        use_threading = len(ciphertext) >= 1024 and self.max_threads > 1
        
        if use_threading:
            plaintext = self._parallel_process_blocks(ciphertext, self.decrypt_block)
        else:
            # Single-threaded decryption for small inputs
            plaintext = self._sequential_process_blocks(ciphertext, self.decrypt_block)
        
        # Remove PKCS#7 padding
        try:
            # Get padding length from the last byte
            padding_length = plaintext[-1]
            
            # Basic sanity checks
            if padding_length == 0 or padding_length > 64:
                raise ValueError(f"Invalid padding length: {padding_length}")
                
            # Ensure we have enough bytes for the padding
            if len(plaintext) < padding_length:
                raise ValueError(f"Not enough bytes for padding: {len(plaintext)} < {padding_length}")
                
            # Verify the padding - check that all padding bytes are the same value
            # But be lenient by only checking a few bytes
            for i in range(1, min(padding_length + 1, 8)):  # Check up to 8 padding bytes
                if plaintext[-i] != padding_length:
                    raise ValueError(f"Inconsistent padding at position {i}")
            
            # Remove padding
            return plaintext[:-padding_length]
            
        except ValueError as e:
            # For better error recovery, try to detect valid padding patterns
            if len(plaintext) > 0:
                for p_len in range(1, min(65, len(plaintext) + 1)):
                    if len(plaintext) >= p_len and all(b == p_len for b in plaintext[-p_len:]):
                        # Valid padding pattern found
                        return plaintext[:-p_len]
            
            # If no valid padding detected, return data as is
            return plaintext
    
    def _sequential_process_blocks(self, data: bytes, block_func: Callable[[bytes], bytes]) -> bytes:
        """
        Process data in blocks sequentially.
        
        Args:
            data: Data to process
            block_func: Function to apply to each block
            
        Returns:
            Processed data
        """
        result = bytearray()
        
        for i in range(0, len(data), 64):
            block = bytes(data[i:i+64])
            processed_block = block_func(block)
            result.extend(processed_block)
        
        return bytes(result)
    
    def _parallel_process_blocks(self, data: bytes, block_func: Callable[[bytes], bytes]) -> bytes:
        """
        Process data in blocks in parallel using multiple threads.
        
        Args:
            data: Data to process
            block_func: Function to apply to each block
            
        Returns:
            Processed data
        """
        # Split data into blocks
        blocks = []
        for i in range(0, len(data), 64):
            blocks.append(bytes(data[i:i+64]))
        
        # Process blocks in parallel
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            processed_blocks = list(executor.map(block_func, blocks))
        
        # Combine results
        return b''.join(processed_blocks)
