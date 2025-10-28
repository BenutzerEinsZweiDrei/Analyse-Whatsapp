# Contributing to WhatsApp Conversation Analyzer

Thank you for your interest in contributing to WhatsApp Conversation Analyzer! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/Analyse-Whatsapp.git
cd Analyse-Whatsapp
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
pip install ruff black isort pytest pytest-cov
```

5. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

- **Python Version**: Python 3.10+
- **Line Length**: 100 characters maximum
- **Formatting**: Use `black` for code formatting
- **Imports**: Use `isort` for import sorting
- **Linting**: Use `ruff` for linting

Format your code before committing:
```bash
black .
isort .
ruff check . --fix
```

### Code Structure

- **Modularity**: Keep functions small and focused
- **Type Hints**: Add type hints to all function signatures
- **Docstrings**: Document all public functions with docstrings
- **Error Handling**: Use try/except with specific exceptions and logging
- **Logging**: Use the centralized logger (`logging.getLogger("whatsapp_analyzer")`)

Example:
```python
from typing import List
import logging

logger = logging.getLogger("whatsapp_analyzer")

def process_text(text: str, max_length: int = 100) -> List[str]:
    """
    Process text and split into chunks.
    
    Args:
        text: Input text to process
        max_length: Maximum length of each chunk
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    logger.debug(f"Processing text of length {len(text)}")
    # Implementation here
    return chunks
```

### Testing

- **Test Coverage**: Aim for >70% coverage
- **Test Structure**: One test file per module
- **Test Naming**: `test_<function_name>_<scenario>`
- **Assertions**: Use descriptive assertion messages

Run tests:
```bash
# All tests
python -m unittest discover tests/ -v

# Specific test file
python -m unittest tests.test_app_modules -v

# With coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

Example test:
```python
import unittest
from app.core.parser import parse_conversations

class TestParser(unittest.TestCase):
    def test_parse_empty_text(self):
        """Test parsing empty text returns empty list."""
        result = parse_conversations("")
        self.assertEqual(result, [], "Empty text should return empty list")
    
    def test_parse_single_message(self):
        """Test parsing single message."""
        text = "23.01.21, 14:30 - Alice: Hello"
        result = parse_conversations(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (CI, dependencies, etc.)
- `perf`: Performance improvements
- `style`: Code style changes (formatting, etc.)

Examples:
```
feat(parser): add support for iOS WhatsApp export format
fix(sentiment): handle empty text without crashing
refactor(core): extract keyword logic to separate module
docs(readme): update installation instructions
test(emojis): add tests for emoji extraction
chore(ci): update GitHub Actions to Python 3.12
```

### Pull Request Process

1. **Update documentation**: Update README.md if adding features
2. **Add tests**: Ensure new code is tested
3. **Run checks**: Run linting and tests locally before pushing
4. **Write clear description**: Explain what and why
5. **Link issues**: Reference related issues if applicable

Pull Request Template:
```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] All tests pass locally
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass
```

## Security

### API Keys and Secrets

- **Never commit API keys** or secrets to the repository
- Use environment variables or `.streamlit/secrets.toml`
- Add sensitive files to `.gitignore`
- Use `app/config.py` for centralized configuration

### Security Issues

If you discover a security vulnerability:
1. **Do NOT open a public issue**
2. Email the maintainers directly
3. Provide details about the vulnerability
4. Allow time for a fix before public disclosure

## Areas for Contribution

### Good First Issues

- Add more unit tests
- Improve documentation
- Fix typos
- Add type hints to existing code
- Improve error messages

### Feature Requests

Before implementing a new feature:
1. Open an issue to discuss the feature
2. Wait for approval from maintainers
3. Implement with tests and documentation

### Bug Reports

When reporting bugs, include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages (if any)
- Minimal example

## Questions?

- Open a [GitHub Discussion](https://github.com/BenutzerEinsZweiDrei/Analyse-Whatsapp/discussions)
- Check existing [Issues](https://github.com/BenutzerEinsZweiDrei/Analyse-Whatsapp/issues)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉
