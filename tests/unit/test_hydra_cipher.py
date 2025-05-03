#!/usr/bin/env python3
"""
Unit tests for the HYDRA encryption algorithm.
"""

import os
import sys
import unittest

# Add parent directory to path so we can import the src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core import HydraCipher

class TestHydraCipher(unittest.TestCase):
    """Test cases for the HydraCipher class."""

    def test_initialization(self):
        """Test that cipher can be initialized with valid keys."""
        # 256-bit key
        key_256 = os.urandom(32)
        cipher_256 = HydraCipher(key_256)
        self.assertEqual(len(cipher_256.key), 32)
        
        # 512-bit key
        key_512 = os.urandom(64)
        cipher_512 = HydraCipher(key_512)
        self.assertEqual(len(cipher_512.key), 64)
    
    def test_invalid_key_size(self):
        """Test that initialization fails with invalid key sizes."""
        # Too small
        with self.assertRaises(ValueError):
            HydraCipher(os.urandom(16))
        
        # Too large
        with self.assertRaises(ValueError):
            HydraCipher(os.urandom(128))
        
        # Not a multiple of 32 bytes
        with self.assertRaises(ValueError):
            HydraCipher(os.urandom(48))
    
    def test_encrypt_decrypt(self):
        """Test encryption and decryption of data."""
        key = os.urandom(32)
        cipher = HydraCipher(key)
        
        # Test with different message sizes
        for size in [1, 10, 63, 64, 65, 128, 1024]:
            message = os.urandom(size)
            encrypted = cipher.encrypt(message)
            decrypted = cipher.decrypt(encrypted)
            
            self.assertEqual(message, decrypted, f"Failed for message of size {size}")
    
    def test_encrypted_size(self):
        """Test that encrypted data has expected size (with padding)."""
        key = os.urandom(32)
        cipher = HydraCipher(key)
        
        # Test with different message sizes
        for size in [1, 10, 63, 64, 65, 128, 1024]:
            message = os.urandom(size)
            encrypted = cipher.encrypt(message)
            
            # Encrypted size should be a multiple of 64 bytes (the block size)
            self.assertEqual(len(encrypted) % 64, 0, 
                            f"Encrypted size not a multiple of block size for message size {size}")
            
            # Encrypted should be at least as large as original (due to padding)
            self.assertGreaterEqual(len(encrypted), len(message), 
                                   f"Encrypted smaller than original for message size {size}")
    
    def test_different_keys(self):
        """Test that different keys produce different ciphertexts."""
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        
        cipher1 = HydraCipher(key1)
        cipher2 = HydraCipher(key2)
        
        message = b"This is a test message"
        
        encrypted1 = cipher1.encrypt(message)
        encrypted2 = cipher2.encrypt(message)
        
        self.assertNotEqual(encrypted1, encrypted2, "Different keys produced identical ciphertexts")
    
    def test_encrypt_block(self):
        """Test encryption of a single block."""
        key = os.urandom(32)
        cipher = HydraCipher(key)
        
        # A single 64-byte block
        block = os.urandom(64)
        
        encrypted_block = cipher.encrypt_block(block)
        self.assertEqual(len(encrypted_block), 64, "Encrypted block size incorrect")
        
        # Test that encryption is deterministic for same input and key
        encrypted_again = cipher.encrypt_block(block)
        self.assertEqual(encrypted_block, encrypted_again, 
                        "Encryption not deterministic for same input and key")
    
    def test_decrypt_block(self):
        """Test decryption of a single block."""
        key = os.urandom(32)
        cipher = HydraCipher(key)
        
        # A single 64-byte block
        block = os.urandom(64)
        
        encrypted_block = cipher.encrypt_block(block)
        decrypted_block = cipher.decrypt_block(encrypted_block)
        
        self.assertEqual(block, decrypted_block, "Block decryption failed")
    
    def test_invalid_block_size(self):
        """Test that block operations fail with invalid block sizes."""
        key = os.urandom(32)
        cipher = HydraCipher(key)
        
        # Too small
        with self.assertRaises(ValueError):
            cipher.encrypt_block(os.urandom(32))
        
        # Too large
        with self.assertRaises(ValueError):
            cipher.encrypt_block(os.urandom(128))
        
        # For decryption
        with self.assertRaises(ValueError):
            cipher.decrypt_block(os.urandom(32))
    
    def test_wrong_key_decryption(self):
        """Test decryption with wrong key."""
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        
        cipher1 = HydraCipher(key1)
        cipher2 = HydraCipher(key2)
        
        message = b"This is a test message"
        
        encrypted = cipher1.encrypt(message)
        
        # Decryption with wrong key should produce different result
        decrypted = cipher2.decrypt(encrypted)
        self.assertNotEqual(message, decrypted, "Decryption with wrong key produced original message")
    
    def test_corrupted_data_decryption(self):
        """Test decryption with corrupted data."""
        key = os.urandom(32)
        cipher = HydraCipher(key)
        
        message = b"This is a test message"
        encrypted = cipher.encrypt(message)
        
        # Corrupt the data
        corrupted = bytearray(encrypted)
        corrupted[0] ^= 0xFF  # Flip bits in the first byte
        
        # Should raise ValueError due to padding validation failure
        with self.assertRaises(ValueError):
            cipher.decrypt(bytes(corrupted))
    
    def test_round_trip_with_unicode(self):
        """Test encryption and decryption with Unicode text."""
        key = os.urandom(32)
        cipher = HydraCipher(key)
        
        text = "Hello, 世界! This includes unicode characters."
        message = text.encode('utf-8')
        
        encrypted = cipher.encrypt(message)
        decrypted = cipher.decrypt(encrypted)
        
        self.assertEqual(message, decrypted, "Unicode round-trip failed")
        self.assertEqual(text, decrypted.decode('utf-8'), "Unicode decoding failed")

if __name__ == '__main__':
    unittest.main()
