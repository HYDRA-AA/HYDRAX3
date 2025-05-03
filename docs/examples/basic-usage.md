# HYDRA Encryption: Basic Usage Examples

This document provides basic examples of how to use the HYDRA encryption algorithm in your projects.

## Prerequisites

- Python 3.6 or higher
- HYDRA encryption library installed

## Basic Encryption and Decryption

### Simple Message Encryption

```python
from hydra.core import HydraCipher
import os

# Generate a secure random key (256 bits / 32 bytes)
key = os.urandom(32)

# Create cipher instance
cipher = HydraCipher(key)

# Message to encrypt
message = b"This is a confidential message."

# Encrypt the message
encrypted = cipher.encrypt(message)

# Decrypt the message
decrypted = cipher.decrypt(encrypted)

# Verify decryption was successful
assert decrypted == message
print("Original:", message.decode('utf-8'))
print("Encrypted (hex):", encrypted.hex())
print("Decrypted:", decrypted.decode('utf-8'))
```

### Working with Text in Different Encodings

```python
from hydra.core import HydraCipher
import os

# Generate a key
key = os.urandom(32)
cipher = HydraCipher(key)

# Working with UTF-8 encoded text
text = "Hello, 世界! This includes unicode characters."
encoded_text = text.encode('utf-8')  # Convert string to bytes

# Encrypt and decrypt
encrypted = cipher.encrypt(encoded_text)
decrypted = cipher.decrypt(encrypted)

# Convert bytes back to string
original_text = decrypted.decode('utf-8')
print(original_text)  # Should print the original message
```

## File Encryption

### Encrypting a Single File

```python
from hydra.core import HydraCipher
import os

def encrypt_file(input_path, output_path, key):
    """Encrypt a file using HYDRA."""
    cipher = HydraCipher(key)
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        data = f_in.read()
        encrypted = cipher.encrypt(data)
        f_out.write(encrypted)
    
    print(f"File encrypted: {input_path} -> {output_path}")

def decrypt_file(input_path, output_path, key):
    """Decrypt a file using HYDRA."""
    cipher = HydraCipher(key)
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        data = f_in.read()
        decrypted = cipher.decrypt(data)
        f_out.write(decrypted)
    
    print(f"File decrypted: {input_path} -> {output_path}")

# Usage example
key = os.urandom(32)  # Generate a random 256-bit key
encrypt_file('document.txt', 'document.encrypted', key)
decrypt_file('document.encrypted', 'document.decrypted.txt', key)
```

### Encrypting Large Files in Chunks

```python
from hydra.core import HydraCipher
import os

def encrypt_large_file(input_path, output_path, key, chunk_size=1024*1024):
    """Encrypt a large file in chunks to avoid memory issues."""
    cipher = HydraCipher(key)
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            
            encrypted_chunk = cipher.encrypt(chunk)
            f_out.write(encrypted_chunk)
    
    print(f"Large file encrypted: {input_path} -> {output_path}")

def decrypt_large_file(input_path, output_path, key, chunk_size=1024*1024):
    """Decrypt a large file in chunks."""
    cipher = HydraCipher(key)
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        while True:
            # Note: chunk_size must be a multiple of 64 (HYDRA's block size)
            # plus potential padding
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            
            decrypted_chunk = cipher.decrypt(chunk)
            f_out.write(decrypted_chunk)
    
    print(f"Large file decrypted: {input_path} -> {output_path}")

# Usage example
key = os.urandom(32)
encrypt_large_file('large_video.mp4', 'large_video.encrypted', key)
decrypt_large_file('large_video.encrypted', 'large_video.decrypted.mp4', key)
```

## Key Management Examples

### Storing and Loading Keys

```python
from hydra.core import HydraCipher
import os
import base64

def generate_and_save_key(key_path, key_size=32):
    """Generate a new key and save it to a file."""
    key = os.urandom(key_size)  # 256-bit (32 bytes) or 512-bit (64 bytes)
    
    # Encode key as base64 for storage
    key_b64 = base64.b64encode(key).decode('utf-8')
    
    with open(key_path, 'w') as f:
        f.write(key_b64)
    
    print(f"Key generated and saved to {key_path}")
    return key

def load_key(key_path):
    """Load a key from a file."""
    with open(key_path, 'r') as f:
        key_b64 = f.read().strip()
    
    # Decode from base64
    key = base64.b64decode(key_b64)
    return key

# Usage example
key = generate_and_save_key('encryption.key')
loaded_key = load_key('encryption.key')

# Verify keys match
assert key == loaded_key
```

### Deriving Keys from Passwords

```python
from hydra.core import HydraCipher
import hashlib
import os

def derive_key_from_password(password, salt=None, key_size=32):
    """Derive a key from a password using PBKDF2."""
    if salt is None:
        salt = os.urandom(16)  # Generate a random salt
    
    # Use PBKDF2 with HMAC-SHA256 and 100,000 iterations (adjust as needed)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations=100000,
        dklen=key_size
    )
    
    return key, salt

# Usage example
password = "s3cur3-p@ssw0rd!"
key, salt = derive_key_from_password(password)

# Store the salt with the encrypted data
cipher = HydraCipher(key)
message = b"Secret message"
encrypted = cipher.encrypt(message)

# Later, to decrypt:
same_key, _ = derive_key_from_password(password, salt)
cipher = HydraCipher(same_key)
decrypted = cipher.decrypt(encrypted)
```

## Error Handling

### Handling Common Errors

```python
from hydra.core import HydraCipher
import os

def safe_encrypt(data, key):
    """Encrypt data with error handling."""
    try:
        cipher = HydraCipher(key)
        return cipher.encrypt(data)
    except ValueError as e:
        print(f"Encryption error: {e}")
        if len(key) not in (32, 64):
            print("Key must be either 32 bytes (256 bits) or 64 bytes (512 bits)")
        return None

def safe_decrypt(encrypted_data, key):
    """Decrypt data with error handling."""
    try:
        cipher = HydraCipher(key)
        return cipher.decrypt(encrypted_data)
    except ValueError as e:
        print(f"Decryption error: {e}")
        return None

# Usage example
key = os.urandom(32)
data = b"Test message"

# Successful case
encrypted = safe_encrypt(data, key)
decrypted = safe_decrypt(encrypted, key)

# Error cases
bad_key = os.urandom(16)  # Wrong key size
result1 = safe_encrypt(data, bad_key)  # Should print error

if encrypted:
    wrong_key = os.urandom(32)  # Different key
    result2 = safe_decrypt(encrypted, wrong_key)  # Will decrypt but with wrong data
    
    # Corrupted data
    corrupted = bytearray(encrypted)
    corrupted[0] ^= 0xFF  # Flip bits in the first byte
    result3 = safe_decrypt(bytes(corrupted), key)  # Should fail due to padding error
```

## Advanced Usage

### Custom Parameters

```python
from hydra.core import HydraCipher
import os

class EnhancedHydraCipher(HydraCipher):
    """HYDRA cipher with enhanced security parameters."""
    
    # Increase minimum rounds for additional security margin
    MIN_ROUNDS = 20   # Default is 16
    
    # Increase maximum rounds for additional security margin
    MAX_ROUNDS = 32   # Default is 24
    
    # Increase number of S-boxes for more substitution variety
    NUM_S_BOXES = 32  # Default is 16

# Usage example
key = os.urandom(32)
cipher = EnhancedHydraCipher(key)

message = b"This uses enhanced security parameters."
encrypted = cipher.encrypt(message)
decrypted = cipher.decrypt(encrypted)

assert decrypted == message
```

## Performance Benchmarking

```python
from hydra.core import HydraCipher
import os
import time

def benchmark_encryption(data_size, key_size=32, iterations=10):
    """Benchmark HYDRA encryption performance."""
    key = os.urandom(key_size)
    cipher = HydraCipher(key)
    
    # Create test data
    data = os.urandom(data_size)
    
    # Benchmark encryption
    start_time = time.time()
    for _ in range(iterations):
        cipher.encrypt(data)
    end_time = time.time()
    
    encryption_time = (end_time - start_time) / iterations
    encryption_speed = data_size / encryption_time / (1024 * 1024)  # MB/s
    
    print(f"Data size: {data_size / 1024:.1f} KB")
    print(f"Average encryption time: {encryption_time:.4f} seconds")
    print(f"Encryption speed: {encryption_speed:.2f} MB/s")

# Run benchmarks for different data sizes
benchmark_encryption(10 * 1024)      # 10 KB
benchmark_encryption(100 * 1024)     # 100 KB
benchmark_encryption(1024 * 1024)    # 1 MB
```

---

**Note**: Remember that HYDRA is an experimental encryption algorithm and should not be used for sensitive applications until it has received extensive review and cryptanalysis from the security community.
