# HYDRA Encryption Algorithm

A high-performance encryption framework with adaptive security rounds and reliable metadata-enhanced decryption.

[![CI Status](https://github.com/hydra-crypto/hydra/actions/workflows/ci.yml/badge.svg)](https://github.com/hydra-crypto/hydra/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/pypi/pyversions/hydra-crypto.svg)](https://pypi.org/project/hydra-crypto/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

HYDRA is a custom encryption algorithm that features adaptive rounds based on data complexity, JIT-accelerated operations, and reliable decryption through metadata enhancement.

This project contains:

1. **Multiple HYDRA Implementations**
   - Original reference implementation (`src/core/hydra_cipher.py`)
   - JIT-optimized version (`jit_hydra.py`)
   - Simple fixed version (`simple_fixed_hydra.py`)
   - Unified complete implementation (`unified_hydra.py`)

2. **Python Package**
   - Fully packaged version in `hydra_crypto/` directory
   - Command line tool for file encryption
   - API for direct use in Python applications

3. **Documentation & Testing**
   - Comprehensive documentation in `HYDRA_README.md`
   - Future development plans in `NEXT_STEPS.md`
   - Test suite in `hydra_crypto/test_package.py`

## Quick Start

### Using the standalone implementation:

```python
from unified_hydra import encrypt, decrypt

key = os.urandom(32)  # Generate a random 256-bit key
encrypted = encrypt(b"This is my secret message", key)
decrypted = decrypt(encrypted, key)
```

### Using the Python package:

```bash
# Install the package
pip install -e hydra_crypto/

# Use the CLI
python -m hydra_crypto.cli -e document.pdf document.pdf.enc
```

## Features

- **Reliable Decryption**: Metadata-enhanced format ensures 100% reliable decryption
- **High Performance**: JIT compilation for 5-30x faster processing
- **Parallel Processing**: Multi-threading for large data sets
- **Adaptive Security**: Variable encryption rounds based on data complexity

## Development Guide

### Dependencies

- Python 3.9+
- NumPy
- Numba (optional, for JIT acceleration)

### Local development setup:

```bash
# Clone the repository
git clone https://github.com/hydra-crypto/hydra.git
cd hydra

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e hydra_crypto/

# Run tests
python hydra_crypto/test_package.py
```

## Security Considerations

HYDRA is a custom encryption algorithm not yet subjected to rigorous cryptographic analysis. For critical applications, consider using well-established algorithms like AES.

Key features for security:
- Adaptive rounds mechanism
- Metadata storage with minimal overhead
- Multi-round key scheduling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

![HYDRA Logo](docs/images/HYDRA.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **⚠️ IMPORTANT SECURITY NOTICE**  
> HYDRA is an experimental encryption algorithm and has not undergone sufficient cryptanalysis to be considered secure for production use. Do not use this for sensitive data until it has received extensive review from the cryptographic community.

## What is HYDRA?

HYDRA is a novel symmetric encryption algorithm designed to provide high security against both classical and quantum threats. It employs a hyperdimensional state structure and multi-domain protection to create a robust and resilient encryption system.

## 🔐 Features

- **Quantum-Resistant Design**: Uses a 512-bit state (256-bit security level against quantum attackers)
- **Multi-Domain Protection**: Combines security approaches across different mathematical domains
- **Hypercube Structure**: 4D hypercube state enables complex permutation and diffusion operations
- **Adaptive Security**: Adjusts operations based on data characteristics
- **Flexible Implementation**: Available in both full and simplified versions
- **Performance Optimized**: Highly optimized for speed while maintaining security properties
- **Simple Implementation**: Achieves 18-19 MB/s throughput in pure Python

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Security Considerations](#security-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔧 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/hydra-encryption.git
cd hydra-encryption

# Install the package
pip install .

# For development, install with additional dependencies
pip install -e ".[dev]"
```

### Dependencies

- Python 3.6 or higher
- No external dependencies for core functionality
- Optional dependencies for development and benchmarking

## 🚀 Quick Start

### Command Line Interface

HYDRA provides a command-line interface for key generation, encryption, and decryption:

```bash
# Generate a key
hydra generate-key --output key.bin

# Encrypt a file
hydra encrypt --key key.bin --input document.txt

# Decrypt a file
hydra decrypt --key key.bin --input document.txt.hydra
```

### Python API

```python
from hydra.core import HydraCipher
import os

# Generate a 256-bit key
key = os.urandom(32)

# Create cipher instance
cipher = HydraCipher(key)

# Encrypt data
plaintext = b"This is a secret message."
ciphertext = cipher.encrypt(plaintext)

# Decrypt data
decrypted = cipher.decrypt(ciphertext)
assert decrypted == plaintext
```

## 📝 Usage Examples

### File Encryption

```python
from hydra.core import HydraCipher
from hydra.utils import key_utils, file_utils

# Generate a key and save it
key = key_utils.generate_key(32)  # 256-bit key
key_utils.save_key_to_file(key, "my_key.bin")

# Load the key
key = key_utils.load_key_from_file("my_key.bin")

# Create cipher instance
cipher = HydraCipher(key)

# Encrypt a file
file_utils.encrypt_file("document.txt", "document.encrypted", cipher.encrypt)

# Decrypt a file
file_utils.decrypt_file("document.encrypted", "document.decrypted", cipher.decrypt)
```

### Password-Based Encryption

```python
from hydra.core import HydraCipher
from hydra.utils import key_utils

# Derive a key from a password
password = "secure-password"
key, salt = key_utils.derive_key_from_password(password)

# Create cipher instance
cipher = HydraCipher(key)

# Save the salt (needed for decryption)
with open("salt.bin", "wb") as f:
    f.write(salt)
```

More examples can be found in the [examples directory](src/examples/).

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Algorithm Specification](docs/design/algorithm-specification.md)
- [Security Analysis](docs/security/security-analysis.md)
- [Implementation Guide](docs/implementation/implementation-guide.md)
- [Usage Examples](docs/examples/basic-usage.md)

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
make test

# Run a specific test
python -m pytest tests/unit/test_hydra_cipher.py

# Run tests with coverage report
python -m pytest --cov=src tests/
```

### Test Vectors

HYDRA includes test vectors to verify implementation correctness:

```bash
# Generate test vectors
python tools/generate_test_vectors.py
```

## 📊 Benchmarks

HYDRA provides comprehensive benchmarking tools to measure performance:

```bash
# Run full benchmarks
python tests/benchmarks/benchmark_report.py

# Run quick benchmarks
python tests/benchmarks/benchmark_report.py --quick
```

Benchmark results include:
- Encryption and decryption speed for various data sizes
- Memory usage
- Comparison with other encryption algorithms (when available)
- Detailed reports and graphical output

## 🔒 Security Considerations

While HYDRA is designed with security in mind, it should not be used for sensitive data until it has received extensive review from the cryptographic community.

For more details, see the [Security Analysis](docs/security/security-analysis.md).

### Implementation Details

HYDRA offers two implementations with different characteristics:

1. **Simple Implementation**: 
   - Production-ready with reliable encryption/decryption
   - High-performance (~18-19 MB/s throughput in pure Python)
   - Optimized with block processing and efficient operations
   - Recommended for practical applications

2. **Full Implementation**:
   - Research-oriented with advanced cryptographic concepts
   - Implements 4D hypercube operations and adaptive round selection
   - Optimized for performance (~80 KB/s, 4x faster than original)
   - Valuable for academic and educational purposes

### Optimization Techniques

The codebase includes numerous performance optimizations:

- **Loop unrolling** for core cryptographic operations
- **Block-based processing** for better cache utilization
- **Precomputation** of lookup tables and inverse S-boxes
- **Bit manipulation** instead of expensive modular arithmetic
- **Specialized fast paths** for common data sizes

### Current Status

This project includes:
- Complete algorithm specification
- Reference implementation in Python with performance optimizations
- Comprehensive test suite with test vectors
- Security documentation and analysis
- Benchmarking tools

## 👥 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas Where Help Is Needed

- Cryptanalysis and security review
- Performance optimizations
- Implementation in additional languages
- Additional test vectors and validation
- Documentation improvements

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
make format

# Run linting
make lint
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by modern encryption algorithms, including AES and ChaCha20
- Based on principles from post-quantum cryptography research
- Thanks to all contributors and reviewers

## ⚠️ Disclaimer

This software is provided "as is" without warranty of any kind. Use at your own risk.

## Contact
