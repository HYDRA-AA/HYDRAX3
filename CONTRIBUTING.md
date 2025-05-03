# Contributing to HYDRA Encryption

Thank you for your interest in contributing to the HYDRA encryption project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Ways to Contribute

There are many ways to contribute to HYDRA:

1. **Code Contributions**: Implement new features, fix bugs, or improve performance
2. **Cryptanalysis**: Analyze the security properties of the algorithm
3. **Documentation**: Improve or expand documentation, examples, and tutorials
4. **Testing**: Create test vectors, improve test coverage, or identify edge cases
5. **Review**: Review pull requests and provide feedback
6. **Bug Reports**: Report bugs or security vulnerabilities (see [SECURITY.md](SECURITY.md) for reporting security issues)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a branch** for your changes
4. **Make your changes** following the coding standards
5. **Add or update tests** as necessary
6. **Run the tests** to ensure they pass
7. **Commit your changes** with clear, descriptive commit messages
8. **Push your changes** to your fork
9. **Submit a pull request** to the main repository

## Pull Request Process

1. Ensure your code follows the project's coding standards
2. Update documentation as necessary
3. Include tests for new functionality
4. Ensure all tests pass
5. Link any related issues in your pull request description
6. Be responsive to feedback and questions

## Coding Standards

- Follow the existing code style in the project
- Write clear, readable, and maintainable code
- Include comments for complex logic or algorithms
- Adhere to language-specific best practices

### Python Specific Guidelines

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Document functions and classes with docstrings

### C/C++ Specific Guidelines

- Follow the project's existing C/C++ style
- Avoid unsafe functions and constructs
- Check for memory leaks and buffer overflows
- Use consistent naming conventions

## Cryptographic Considerations

When contributing to cryptographic code:

- **Do not** invent your own cryptographic primitives without extensive review
- **Do** follow established cryptographic best practices
- **Document** security assumptions and limitations
- **Consider** side-channel attack vectors in implementations

## Development Environment

### Setting Up

```bash
# Clone your fork
git clone https://github.com/yourusername/hydra-encryption.git
cd hydra-encryption

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_cipher.py

# Run with coverage
pytest --cov=hydra
```

## Documentation Guidelines

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include examples where appropriate
- Explain cryptographic concepts for educational purposes

## Review Process

All submissions require review before being merged:

1. Automated tests must pass
2. At least one maintainer must approve the changes
3. Security-related changes require additional review
4. Significant changes to the core algorithm require cryptographic review

## Questions?

If you have questions about contributing, please open a GitHub issue labeled "question".

Thank you for contributing to making encryption more secure and accessible!
