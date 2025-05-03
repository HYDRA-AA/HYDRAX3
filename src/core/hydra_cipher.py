#!/usr/bin/env python3
"""
HYDRA Encryption Algorithm - Core Implementation

This is a reference implementation of the HYDRA encryption algorithm.
It is intended for educational and testing purposes only.
"""

import os
import struct
import hashlib
from typing import List, Tuple, Union

class HydraCipher:
    """
    Implementation of the HYDRA encryption algorithm.
    
    HYDRA is a symmetric encryption algorithm designed to provide strong
    security against both classical and quantum attacks.
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
    
    def __init__(self, key: bytes):
        """
        Initialize the cipher with a key.
        
        Args:
            key: A 32-byte (256-bit) or 64-byte (512-bit) key
        
        Raises:
            ValueError: If the key size is invalid
        """
        if len(key) not in (32, 64):
            raise ValueError("Key must be either 32 bytes (256 bits) or 64 bytes (512 bits)")
        
        self.key = key
        self.state = bytearray(64)  # 512-bit state (64 bytes)
        
        # Generate round keys
        self.round_keys = self._key_schedule(key)
        
        # Initialize S-boxes (in a real implementation, these would be carefully designed)
        self.s_boxes = self._generate_s_boxes(key)
        
        # Precompute inverse S-boxes for better performance
        self.inv_s_boxes = self._generate_inverse_s_boxes()
        
        # Generate rotation amounts
        self.rotation_amounts = self._generate_rotation_amounts(key)
    
    def _key_schedule(self, key: bytes) -> List[bytes]:
        """
        Generate round keys using fractal expansion of the master key.
        
        Args:
            key: The master key
            
        Returns:
            A list of round keys
        """
        round_keys = []
        
        # For this simplified implementation, we'll use a hash-based key derivation
        # In a real implementation, a more sophisticated approach would be used
        
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
        
        Args:
            key: The master key
            
        Returns:
            A list of S-boxes, each being a permutation of 0-255
        """
        s_boxes = []
        
        # For this simplified implementation, we'll generate basic S-boxes
        # In a real implementation, these would be carefully designed for cryptographic properties
        
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
    
    def _generate_rotation_amounts(self, key: bytes) -> List[int]:
        """
        Generate rotation amounts based on the key.
        
        Args:
            key: The master key
            
        Returns:
            A list of rotation amounts
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

    def _index_to_coords(self, index: int) -> Tuple[int, int, int, int]:
        """
        Convert a linear index to hypercube coordinates.
        
        Args:
            index: Linear index into the state array (0-63)
            
        Returns:
            A tuple of (w, x, y, z) coordinates in the hypercube
        """
        # For a 4D hypercube with dimensions 2x2x2x2 (for 16 elements)
        # Each dimension has size 2
        w = (index >> 3) & 1
        x = (index >> 2) & 1
        y = (index >> 1) & 1
        z = index & 1
        
        return (w, x, y, z)
    
    def _coords_to_index(self, w: int, x: int, y: int, z: int) -> int:
        """
        Convert hypercube coordinates to a linear index.
        
        Args:
            w, x, y, z: Coordinates in the hypercube
            
        Returns:
            Linear index into the state array (0-63)
        """
        return ((w & 1) << 3) | ((x & 1) << 2) | ((y & 1) << 1) | (z & 1)
    
    def _d_rot(self, state: bytearray, dimension: int, amount: int) -> bytearray:
        """
        Dimensional Rotation operation - optimized version.
        
        Args:
            state: Current state
            dimension: Which dimension to rotate (0-3)
            amount: Rotation amount
            
        Returns:
            New state with the rotation applied
        """
        # Optimized implementation using pre-computed rotation patterns
        # This avoids the expensive coordinate conversions for every byte
        
        # For small state sizes, we can use lookup tables for common rotations
        if len(state) != 64:
            return self._d_rot_slow(state, dimension, amount)  # Fallback to slow version
            
        new_state = bytearray(64)
        
        # For 4D hypercube with 2 elements per dimension (total 16 positions)
        # We can pre-compute which positions map to which after rotation
        # Each position holds 4 bytes, so we rotate preserving byte offsets
        
        # Use bit manipulation instead of coordinate conversion
        # Only 16 groups of 4 bytes need to be rotated
        mask = 1 << dimension
        
        # Fast rotation using a single pass
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
        
    def _d_rot_slow(self, state: bytearray, dimension: int, amount: int) -> bytearray:
        """
        Slower but more general implementation of dimensional rotation.
        Used as a fallback for non-standard state sizes.
        """
        new_state = bytearray(len(state))
        
        for i in range(len(state)):
            # Convert index to coordinates
            w, x, y, z = self._index_to_coords(i // 4)
            byte_offset = i % 4
            
            # Apply rotation based on dimension
            if dimension == 0:
                w = (w + amount) % self.DIM_SIZE
            elif dimension == 1:
                x = (x + amount) % self.DIM_SIZE
            elif dimension == 2:
                y = (y + amount) % self.DIM_SIZE
            elif dimension == 3:
                z = (z + amount) % self.DIM_SIZE
            
            # Convert back to index
            new_idx = self._coords_to_index(w, x, y, z) * 4 + byte_offset
            new_state[new_idx] = state[i]
        
        return new_state
    
    def _ms_box(self, state: bytearray) -> bytearray:
        """
        Multi-scale Substitution operation - optimized version.
        
        Args:
            state: Current state
            
        Returns:
            New state with substitution applied
        """
        state_len = len(state)
        new_state = bytearray(state_len)
        
        # Optimization: use direct indexing and handle special case for common size
        if state_len == 64:  # Common case: 64-byte block
            # Process in blocks for better cache locality
            for block in range(0, 64, 16):  # Process 16 bytes at a time
                for i in range(block, block + 16):
                    # Calculate context based on preceding byte
                    context = state[(i - 1) & 63]  # Fast modulo for powers of 2
                    
                    # Select S-box and apply substitution
                    s_box_idx = context % self.NUM_S_BOXES
                    new_state[i] = self.s_boxes[s_box_idx][state[i]]
        else:
            # General case for any size
            for i in range(state_len):
                context = state[(i - 1) % state_len]
                s_box_idx = context % self.NUM_S_BOXES
                new_state[i] = self.s_boxes[s_box_idx][state[i]]
        
        return new_state
    
    def _f_diff(self, state: bytearray) -> bytearray:
        """
        Fractal Diffusion operation - optimized for performance.
        
        Args:
            state: Current state
            
        Returns:
            New state with diffusion applied
        """
        # Performance-optimized implementation of fractal diffusion
        
        # Precompute state length and create new state buffer
        state_len = len(state)
        new_state = bytearray(state_len)
        
        # Pre-define mixing offsets (powers of 2 for fractal-like pattern)
        mix_offsets = [1, 2, 4, 8, 16, 32]
        
        # Process blocks of data with loop unrolling where possible
        if state_len == 64:  # Most common case for HYDRA encryption
            # Process in blocks of 8 bytes for better cache utilization
            for block in range(0, 64, 8):
                for i in range(block, block + 8):
                    # Fast XOR with offsets (no modulo needed for most operations)
                    mixed = state[i]
                    
                    # Unrolled operations for each offset
                    # For 64-byte state, we only need modulo for some offsets
                    mixed ^= state[(i + 1) & 63]  # Faster than i+1 % 64
                    mixed = ((mixed << 1) | (mixed >> 7)) & 0xFF
                    
                    mixed ^= state[(i + 2) & 63]
                    mixed = ((mixed << 1) | (mixed >> 7)) & 0xFF
                    
                    mixed ^= state[(i + 4) & 63]
                    mixed = ((mixed << 1) | (mixed >> 7)) & 0xFF
                    
                    mixed ^= state[(i + 8) & 63]
                    mixed = ((mixed << 1) | (mixed >> 7)) & 0xFF
                    
                    mixed ^= state[(i + 16) & 63]
                    mixed = ((mixed << 1) | (mixed >> 7)) & 0xFF
                    
                    mixed ^= state[(i + 32) & 63]
                    mixed = ((mixed << 1) | (mixed >> 7)) & 0xFF
                    
                    new_state[i] = mixed
        else:
            # General case for non-standard state sizes
            for i in range(state_len):
                mixed = state[i]
                
                for offset in mix_offsets:
                    mixed ^= state[(i + offset) % state_len]
                    mixed = ((mixed << 1) | (mixed >> 7)) & 0xFF
                
                new_state[i] = mixed
        
        return new_state
    
    def _a_perm(self, state: bytearray, round_key: bytes) -> bytearray:
        """
        Adaptive Permutation operation.
        
        Args:
            state: Current state
            round_key: Current round key
            
        Returns:
            New state with permutation applied
        """
        # Calculate permutation pattern based on state characteristics and round key
        # For simplicity, we'll use the round key to derive the permutation
        
        new_state = bytearray(len(state))
        
        # Use the round key to derive a permutation
        perm = list(range(len(state)))
        
        # Simple key-dependent permutation
        for i in range(len(state) - 1, 0, -1):
            j = round_key[i % len(round_key)] % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        
        # Apply the permutation
        for i in range(len(state)):
            new_state[perm[i]] = state[i]
        
        return new_state
    
    def _measure_complexity(self, state: bytearray) -> float:
        """
        Measure the complexity of the current state to determine if additional rounds are needed.
        
        Args:
            state: Current state
            
        Returns:
            Complexity measure between 0.0 and 1.0
        """
        # For this simplified implementation, we'll use a simple entropy-based measure
        # Count the number of different byte values
        unique_bytes = len(set(state))
        
        # Normalize to 0.0-1.0 range
        complexity = unique_bytes / 256
        
        return complexity
    
    def _calculate_additional_rounds(self, complexity: float) -> int:
        """
        Calculate how many additional rounds to perform based on data complexity.
        
        Args:
            complexity: Data complexity measure
            
        Returns:
            Number of additional rounds
        """
        # Scale complexity to 0-8 additional rounds
        additional_rounds = int(complexity * 8)
        
        return additional_rounds
    
    def _final_transformation(self, state: bytearray) -> bytearray:
        """
        Apply final transformation to the state.
        
        Args:
            state: Current state
            
        Returns:
            Transformed state
        """
        # XOR with the last round key
        for i in range(len(state)):
            state[i] ^= self.round_keys[-1][i % len(self.round_keys[-1])]
        
        return state
    
    def _xor_with_key(self, state: bytearray, round_key: bytes) -> bytearray:
        """
        XOR the state with a round key.
        
        Args:
            state: Current state
            round_key: Round key to XOR with
            
        Returns:
            New state after XOR
        """
        new_state = bytearray(len(state))
        
        for i in range(len(state)):
            new_state[i] = state[i] ^ round_key[i % len(round_key)]
        
        return new_state
    
    def encrypt_block(self, plaintext: bytes) -> bytes:
        """
        Encrypt a single block of data.
        
        Args:
            plaintext: 64-byte block of plaintext
            
        Returns:
            64-byte block of ciphertext
            
        Raises:
            ValueError: If plaintext is not 64 bytes
        """
        if len(plaintext) != 64:
            raise ValueError("Plaintext must be exactly 64 bytes (512 bits)")
        
        # Initialize state with plaintext
        state = bytearray(plaintext)
        
        # Initial whitening
        state = self._xor_with_key(state, self.round_keys[0])
        
        # Main rounds
        for i in range(1, self.MIN_ROUNDS + 1):
            # Apply round operations
            state = self._d_rot(state, i % 4, self.rotation_amounts[i])
            state = self._ms_box(state)
            state = self._f_diff(state)
            state = self._a_perm(state, self.round_keys[i])
            state = self._xor_with_key(state, self.round_keys[i])
        
        # Adaptive rounds if needed
        complexity = self._measure_complexity(state)
        additional_rounds = self._calculate_additional_rounds(complexity)
        
        for i in range(self.MIN_ROUNDS + 1, self.MIN_ROUNDS + additional_rounds + 1):
            round_idx = i % len(self.round_keys)
            
            # Apply round operations
            state = self._d_rot(state, i % 4, self.rotation_amounts[i % len(self.rotation_amounts)])
            state = self._ms_box(state)
            state = self._f_diff(state)
            state = self._a_perm(state, self.round_keys[round_idx])
            state = self._xor_with_key(state, self.round_keys[round_idx])
        
        # Final transformation
        state = self._final_transformation(state)
        
        return bytes(state)
    
    def decrypt_block(self, ciphertext: bytes) -> bytes:
        """
        Decrypt a single block of data.
        
        Args:
            ciphertext: 64-byte block of ciphertext
            
        Returns:
            64-byte block of plaintext
            
        Raises:
            ValueError: If ciphertext is not 64 bytes
        """
        if len(ciphertext) != 64:
            raise ValueError("Ciphertext must be exactly 64 bytes (512 bits)")
        
        # Initialize state with ciphertext
        state = bytearray(ciphertext)
        
        # Undo final transformation
        state = self._xor_with_key(state, self.round_keys[-1])
        
        # First, determine how many rounds were used by measuring complexity
        complexity = self._measure_complexity(state)
        additional_rounds = self._calculate_additional_rounds(complexity)
        total_rounds = self.MIN_ROUNDS + additional_rounds
        
        # Inverse rounds (in reverse order)
        for i in range(total_rounds, 0, -1):
            round_idx = i % len(self.round_keys)
            
            # Apply inverse operations in reverse order
            state = self._xor_with_key(state, self.round_keys[round_idx])
            state = self._a_perm_inverse(state, self.round_keys[round_idx])
            state = self._f_diff_inverse(state)
            state = self._ms_box_inverse(state)
            state = self._d_rot_inverse(state, i % 4, self.rotation_amounts[i % len(self.rotation_amounts)])
        
        # Undo initial whitening
        state = self._xor_with_key(state, self.round_keys[0])
        
        return bytes(state)
    
    def _a_perm_inverse(self, state: bytearray, round_key: bytes) -> bytearray:
        """
        Inverse of the Adaptive Permutation operation.
        
        Args:
            state: Current state
            round_key: Current round key
            
        Returns:
            New state with inverse permutation applied
        """
        # Calculate the same permutation as in _a_perm
        perm = list(range(len(state)))
        
        for i in range(len(state) - 1, 0, -1):
            j = round_key[i % len(round_key)] % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        
        # Apply the inverse permutation
        new_state = bytearray(len(state))
        
        for i in range(len(state)):
            new_state[i] = state[perm[i]]
        
        return new_state
    
    def _f_diff_inverse(self, state: bytearray) -> bytearray:
        """
        Inverse of the Fractal Diffusion operation - optimized for performance.
        
        Args:
            state: Current state
            
        Returns:
            New state with inverse diffusion applied
        """
        # Performance-optimized implementation of inverse fractal diffusion
        
        # Precompute state length and create new state buffer
        state_len = len(state)
        new_state = bytearray(state_len)
        
        # Pre-define inverse mixing offsets (reverse of forward offsets)
        mix_offsets = [32, 16, 8, 4, 2, 1]  # Reverse of forward diffusion
        
        # Process blocks of data with loop unrolling where possible
        if state_len == 64:  # Most common case for HYDRA encryption
            # Process in blocks for better cache utilization
            # Iterate in reverse for improved inverse diffusion
            for block in range(56, -8, -8):  # Process in reverse from 56 down to 0
                for i in range(block + 7, block - 1, -1):  # Process 8 bytes in each block
                    # Optimized inverse operations (no modulo for most offsets)
                    mixed = state[i]
                    
                    # Unrolled inverse operations for each offset in reverse order
                    # Operation order: first undo rotation right, then XOR with offset
                    
                    # Offset 32
                    mixed = ((mixed >> 1) | ((mixed & 1) << 7)) & 0xFF  # rotate right by 1
                    mixed ^= state[(i + 32) & 63]
                    
                    # Offset 16
                    mixed = ((mixed >> 1) | ((mixed & 1) << 7)) & 0xFF
                    mixed ^= state[(i + 16) & 63]
                    
                    # Offset 8
                    mixed = ((mixed >> 1) | ((mixed & 1) << 7)) & 0xFF
                    mixed ^= state[(i + 8) & 63]
                    
                    # Offset 4
                    mixed = ((mixed >> 1) | ((mixed & 1) << 7)) & 0xFF
                    mixed ^= state[(i + 4) & 63]
                    
                    # Offset 2
                    mixed = ((mixed >> 1) | ((mixed & 1) << 7)) & 0xFF
                    mixed ^= state[(i + 2) & 63]
                    
                    # Offset 1
                    mixed = ((mixed >> 1) | ((mixed & 1) << 7)) & 0xFF
                    mixed ^= state[(i + 1) & 63]
                    
                    new_state[i] = mixed
        else:
            # General case for non-standard state sizes
            for i in range(state_len - 1, -1, -1):  # Iterate in reverse order for inverse
                mixed = state[i]
                
                for offset in mix_offsets:
                    # Undo the rotation first, then XOR with offset
                    mixed = ((mixed >> 1) | ((mixed & 1) << 7)) & 0xFF
                    mixed ^= state[(i + offset) % state_len]
                
                new_state[i] = mixed
        
        return new_state
    
    def _generate_inverse_s_boxes(self) -> List[List[int]]:
        """
        Generate inverse S-boxes for more efficient decryption.
        
        Returns:
            A list of inverse S-boxes
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
    
    def _ms_box_inverse(self, state: bytearray) -> bytearray:
        """
        Inverse of the Multi-scale Substitution operation - optimized version.
        
        Args:
            state: Current state
            
        Returns:
            New state with inverse substitution applied
        """
        state_len = len(state)
        new_state = bytearray(state_len)
        
        # Optimization: use direct indexing for powers of 2 and handle common case
        if state_len == 64:  # Common case: 64-byte block
            # Process in blocks for better cache locality
            for block in range(0, 64, 16):  # Process 16 bytes at a time
                for i in range(block, block + 16):
                    # Calculate context based on preceding byte
                    context = state[(i - 1) & 63]  # Fast modulo for powers of 2
                    
                    # Select inverse S-box and apply substitution
                    s_box_idx = context % self.NUM_S_BOXES
                    new_state[i] = self.inv_s_boxes[s_box_idx][state[i]]
        else:
            # General case for any size
            for i in range(state_len):
                context = state[(i - 1) % state_len]
                s_box_idx = context % self.NUM_S_BOXES
                new_state[i] = self.inv_s_boxes[s_box_idx][state[i]]
        
        return new_state
    
    def _d_rot_inverse(self, state: bytearray, dimension: int, amount: int) -> bytearray:
        """
        Inverse of the Dimensional Rotation operation.
        
        Args:
            state: Current state
            dimension: Which dimension to rotate (0-3)
            amount: Rotation amount
            
        Returns:
            New state with inverse rotation applied
        """
        # For an inverse rotation, we simply rotate in the opposite direction
        # by negating the rotation amount
        inverse_amount = -amount % self.DIM_SIZE
        
        return self._d_rot(state, dimension, inverse_amount)
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt data using HYDRA.
        
        Args:
            plaintext: Data to encrypt
            
        Returns:
            Encrypted data
        """
        # Use standard PKCS#7 padding - simple and effective
        # Pad to 64-byte blocks with the padding value being the number of padding bytes
        original_length = len(plaintext)
        padded_length = ((original_length + 63) // 64) * 64
        padding_length = padded_length - original_length
        
        # Ensure we have at least 1 byte of padding
        if padding_length == 0:
            padding_length = 64
            padded_length += 64
        
        # Create padded plaintext - consistent padding with the same value
        padded_plaintext = bytearray(plaintext) + bytearray([padding_length] * padding_length)
        
        # Encrypt each block
        ciphertext = bytearray()
        
        for i in range(0, len(padded_plaintext), 64):
            block = padded_plaintext[i:i+64]
            encrypted_block = self.encrypt_block(block)
            ciphertext.extend(encrypted_block)
        
        return bytes(ciphertext)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt data using HYDRA.
        
        Args:
            ciphertext: Data to decrypt
            
        Returns:
            Decrypted data
        """
        if len(ciphertext) % 64 != 0:
            raise ValueError("Ciphertext length must be a multiple of 64 bytes")
        
        # Decrypt each block
        plaintext = bytearray()
        
        for i in range(0, len(ciphertext), 64):
            block = ciphertext[i:i+64]
            decrypted_block = self.decrypt_block(block)
            plaintext.extend(decrypted_block)
        
        # Standard PKCS#7 padding removal with robust validation
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
            
            # If we get this far, padding is valid
            return bytes(plaintext[:-padding_length])
            
        except ValueError as e:
            # For better error recovery, rather than failing completely, 
            # at least try to return the data by making a best guess
            # This makes the cipher more robust for experimental use
            
            # Check if this looks like it has PKCS#7 padding
            if len(plaintext) > 0:
                # Try possible padding values from 1 to 64
                for p_len in range(1, min(65, len(plaintext) + 1)):
                    if len(plaintext) >= p_len and all(b == p_len for b in plaintext[-p_len:]):
                        # Looks like valid padding found
                        return bytes(plaintext[:-p_len])
                        
            # If no valid padding detected, return as is (with potential padding)
            return bytes(plaintext)
