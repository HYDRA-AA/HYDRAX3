# Contributing to HYDRA

Thank you for your interest in contributing to the HYDRA encryption project! We welcome contributions from everyone.

## Ways to Contribute

There are many ways to contribute to HYDRA:

1. **Report Bugs**: Submit bug reports by creating an issue on GitHub
2. **Request Features**: Suggest enhancements or new features
3. **Improve Documentation**: Help make our documentation clearer and more comprehensive
4. **Submit Code**: Contribute code improvements, fixes, or new features
5. **Review Code**: Help review pull requests and ensure code quality
6. **Security Analysis**: Analyze the algorithm and implementation for security concerns
7. **Cryptanalysis**: Analyze the security properties of the algorithm

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Follow the [code of conduct](CODE_OF_CONDUCT.md) in all interactions.

## Development Process

### Setting up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/hydra.git
   cd hydra
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit them with clear, descriptive messages.
3. Write or update tests as needed.
4. Ensure all tests pass:
   ```bash
   python -m pytest
   ```
5. Update documentation as necessary.

### Submitting Pull Requests

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a pull request from your branch to the main repository.
3. Clearly describe the changes and their purpose.
4. Reference any related issues.
5. Be responsive to feedback and questions.

## Code Standards

- Follow PEP 8 style guidelines for Python code
- Include docstrings and type hints
- Write unit tests for new functionality
- Keep functions focused and modular
- Use descriptive variable and function names
- Add comments for complex sections

### Language-Specific Guidelines

#### Python
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Document functions and classes with docstrings

#### C/C++ (if applicable)
- Follow the project's existing C/C++ style
- Avoid unsafe functions and constructs
- Check for memory leaks and buffer overflows
- Use consistent naming conventions

## Security Considerations

When contributing to cryptographic code:

1. **Avoid Common Pitfalls**: Be aware of timing attacks, side-channel vulnerabilities, and other cryptographic pitfalls.
2. **Randomness is Critical**: Any use of randomness should use cryptographically secure random number generators.
3. **Document Security Assumptions**: Clearly document any security assumptions or limitations.
4. **Report Security Issues Privately**: If you discover a security vulnerability, please report it privately following our [Security Policy](SECURITY.md).
5. **Don't Invent Primitives**: Don't invent your own cryptographic primitives without extensive review.
6. **Follow Best Practices**: Follow established cryptographic best practices.
7. **Consider Side-Channels**: Consider side-channel attack vectors in implementations.

## Review Process

Pull requests will be reviewed by project maintainers. The review process includes:

1. Code quality review
2. Test coverage verification
3. Documentation review
4. Security considerations
5. Automated tests must pass
6. At least one maintainer must approve the changes
7. Security-related changes require additional review
8. Significant changes to the core algorithm require cryptographic review

## Documentation Guidelines

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include examples where appropriate
- Explain cryptographic concepts for educational purposes

## License

By contributing, you agree that your contributions will be licensed under the project's MIT license.

## Questions?

If you have questions or need help, feel free to:

- Open an issue with your question labeled "question"
- Contact the maintainers

Thank you for contributing to HYDRA!
