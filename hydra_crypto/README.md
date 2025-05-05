# HYDRA Cryptography Package

<p align="center">
  <img src="docs/images/HYDRA.png" alt="HYDRA Logo" width="200"/>
</p>

[![PyPI version](https://img.shields.io/pypi/v/hydra-crypto.svg)](https://pypi.org/project/hydra-crypto/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hydra-crypto.svg)](https://pypi.org/project/hydra-crypto/)
[![License](https://img.shields.io/pypi/l/hydra-crypto.svg)](https://github.com/hydra-crypto/hydra/blob/main/LICENSE)

A high-performance, metadata-enhanced encryption library featuring the HYDRA algorithm.

## Features

- **Performance-Optimized**: JIT compilation for maximum speed
- **Reliable Decryption**: Metadata-enhanced format for 100% reliable decryption
- **Parallel Processing**: Multi-threading support for large data
- **Adaptive Security**: Variable encryption rounds based on data complexity
- **Flexible API**: Both high-level and low-level crypto interfaces
- **Command Line Tool**: Ready-to-use file encryption utility

## Installation

```bash
pip install hydra-crypto
```

For development installations:

```bash
git clone https://github.com/hydra-crypto/hydra.git
cd hydra
pip install -e ".[dev]"
```

## Usage Examples

### Basic Encryption/Decryption

```python
from hydra_crypto import encrypt, decrypt, generate_key

# Generate a secure random key
key = generate_key()  # 256-bit key

# Encrypt some data
message = b"This is a secret message"
encrypted = encrypt(message, key)

# Decrypt the data
decrypted = decrypt(encrypted, key)
print(decrypted)  # b'This is a secret message'
```

### File Encryption

```python
from hydra_crypto import encrypt_file, decrypt_file, derive_key_from_password

# Derive a key from a password
password = "my-secure-password"
key, salt = derive_key_from_password(password)

# Encrypt a file
encrypt_file("document.pdf", "document.pdf.enc", key)

# Decrypt a file
decrypt_file("document.pdf.enc", "document-decrypted.pdf", key)
```

### Command Line Usage

The package installs a command-line utility for easy file encryption:

```bash
# Encrypt a file (will prompt for password)
hydra-encrypt -e document.pdf document.pdf.enc

# Decrypt a file
hydra-encrypt -d document.pdf.enc document-restored.pdf

# Provide password directly (less secure)
hydra-encrypt -e document.pdf document.pdf.enc -p mypassword
```

### Advanced Usage with Custom Parameters

```python
from hydra_crypto import UnifiedHydraCipher

# Create a cipher with custom parameters
cipher = UnifiedHydraCipher(
    key=b'my-32-byte-key........................',
    block_size=32,  # Smaller blocks
    min_rounds=2,   # More rounds for added security
    max_additional_rounds=16,  # More adaptive rounds
    max_threads=4   # Control thread usage
)

# Encrypt directly with the cipher
encrypted = cipher.encrypt(b"My data")

# Decrypt with the same parameters
decrypted = cipher.decrypt(encrypted)
```

## Performance Notes

- Small data (< 1KB): Non-parallel mode recommended
- Large data (> 100KB): Parallel mode provides significant speedup
- First encryption with JIT may have compilation overhead
- Subsequent operations benefit from JIT optimization

## Security Considerations

- Key management is crucial - protect your encryption keys
- The algorithm uses an adaptive rounds mechanism based on data complexity
- HYDRA is not a standardized algorithm - use established algorithms (AES, ChaCha20) for critical applications
- Always use password-based key derivation with sufficient iterations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
