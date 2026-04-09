# Contributing to CASMO

Thank you for your interest in contributing to CASMO! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CASMO.git
   cd CASMO
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/abderahmane-ai/CASMO.git
   ```

## Development Setup

### Install in Development Mode

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## How to Contribute

### Reporting Bugs

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include a minimal reproducible example
- Specify your environment (OS, Python version, PyTorch version)
- Include the full error traceback

### Suggesting Features

- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the motivation and use case
- Provide example usage if possible
- Consider if it fits CASMO's scope and philosophy

### Submitting Changes

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Add tests** for new functionality

4. **Run tests** to ensure nothing breaks:
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Code Formatting

```bash
# Format code with Black
black casmo.py tests/ examples/ --line-length 100

# Check with flake8
flake8 casmo.py tests/ examples/ --max-line-length=100
```

### Naming Conventions

- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Documentation

- Use Google-style docstrings
- Include parameter types and return types
- Provide usage examples for complex functions

Example:
```python
def compute_agar(m_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    """
    Compute Adaptive Gradient Alignment Ratio (AGAR).
    
    Args:
        m_t: First moment estimate (mean of gradients)
        v_t: Second moment estimate (mean of squared gradients)
    
    Returns:
        AGAR value in range [0, 1], where 1 indicates pure signal
        and 0 indicates pure noise.
    
    Example:
        >>> m_t = torch.tensor([0.5, 0.3])
        >>> v_t = torch.tensor([0.3, 0.2])
        >>> agar = compute_agar(m_t, v_t)
    """
    signal = m_t ** 2
    noise = v_t - signal
    return signal / (signal + noise + 1e-8)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_optimizer_interface.py -v

# Run with coverage
pytest tests/ --cov=casmo --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test edge cases and error conditions
- Aim for >90% code coverage

Example:
```python
def test_agar_perfect_signal():
    """Test AGAR = 1.0 for zero-variance gradients."""
    param = torch.nn.Parameter(torch.randn(10, 10))
    optimizer = CASMO([param], lr=1e-3)
    
    # Simulate consistent gradients
    for _ in range(100):
        param.grad = torch.ones(10, 10) * 0.5
        optimizer.step()
        optimizer.zero_grad()
    
    # Check AGAR is close to 1.0
    group_state = optimizer._group_states[0]
    assert group_state['current_agar'] > 0.95
```

## Documentation

### Updating Documentation

When adding features or making changes:

1. Update relevant docstrings
2. Update `README.md` if user-facing
3. Update `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/)
4. Add examples to `examples/` if appropriate
5. Update `docs/` if adding major features

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Updated CHANGELOG.md
- [ ] Rebased on latest `main` branch

### PR Guidelines

1. **Title**: Use clear, descriptive titles
   - Good: "Add support for sparse gradients"
   - Bad: "Fix bug"

2. **Description**: Use the PR template
   - Explain what and why
   - Link related issues
   - Describe testing performed

3. **Size**: Keep PRs focused and reasonably sized
   - Large changes should be discussed in an issue first
   - Consider breaking into multiple PRs

4. **Review**: Be responsive to feedback
   - Address all comments
   - Ask questions if unclear
   - Update PR based on feedback

### After Submission

- CI checks must pass
- At least one maintainer approval required
- Maintainers may request changes
- Once approved, maintainers will merge

## Development Workflow

### Syncing with Upstream

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your local main
git checkout main
git merge upstream/main

# Update your feature branch
git checkout feature/your-feature-name
git rebase main
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: Add support for learning rate warmup

Implements linear warmup for the first N steps, which helps
stabilize training in some scenarios.

Closes #123
```

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the [documentation](docs/)
- Review [examples](examples/)

## Recognition

Contributors will be recognized in:
- `CHANGELOG.md` for their contributions
- GitHub contributors page
- Release notes for significant contributions

Thank you for contributing to CASMO! 🎉
