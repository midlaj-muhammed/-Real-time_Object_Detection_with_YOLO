# Contributing to Real-time Object Detection with YOLO

Thank you for your interest in contributing to this project! We welcome contributions from the community and are grateful for your support.

## 🤝 How to Contribute

### 🐛 Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Use the issue templates** when available
3. **Provide detailed information** including:
   - Operating system and version
   - Python version
   - Error messages and full stack traces
   - Steps to reproduce the issue
   - Expected vs actual behavior

### 💡 Suggesting Features

We welcome feature suggestions! Please:

1. **Check existing feature requests** first
2. **Describe the feature** in detail
3. **Explain the use case** and benefits
4. **Consider implementation complexity**

### 🔧 Code Contributions

#### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Real-time_Object_Detection_with_YOLO.git
   cd Real-time_Object_Detection_with_YOLO
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

#### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request** on GitHub

#### Pull Request Guidelines

- **Use descriptive titles** and detailed descriptions
- **Reference related issues** using `#issue-number`
- **Include screenshots** for UI changes
- **Update documentation** as needed
- **Ensure all tests pass**
- **Keep changes focused** - one feature per PR

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** where appropriate
- Write **docstrings** for functions and classes
- Keep **line length** under 88 characters
- Use **meaningful variable names**

### Code Quality

- **Write tests** for new features
- **Maintain test coverage** above 80%
- **Use descriptive commit messages**
- **Comment complex logic**
- **Remove unused imports and variables**

### Example Code Style

```python
def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Perform object detection on an image.
    
    Args:
        image (np.ndarray): Input image in BGR format
        confidence_threshold (float): Minimum confidence for detections
        
    Returns:
        List[Dict[str, Any]]: List of detected objects with their properties
    """
    if self.model is None:
        logger.error("Model not loaded")
        return []
    
    # Implementation here...
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_detector.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Write tests for **all new functions**
- Use **descriptive test names**
- Test **edge cases** and **error conditions**
- Mock **external dependencies**

## 📚 Documentation

### Updating Documentation

- Update **README.md** for user-facing changes
- Update **docstrings** for code changes
- Add **examples** for new features
- Update **configuration** documentation

### Documentation Style

- Use **clear, concise language**
- Include **code examples**
- Add **screenshots** for UI features
- Keep **formatting consistent**

## 🎯 Areas for Contribution

### High Priority
- **Performance optimization**
- **Bug fixes**
- **Documentation improvements**
- **Test coverage**

### Medium Priority
- **New YOLO model support**
- **UI/UX enhancements**
- **Additional export formats**
- **Mobile responsiveness**

### Low Priority
- **Code refactoring**
- **Additional examples**
- **Internationalization**
- **Advanced features**

## 📞 Getting Help

If you need help with contributing:

1. **Check the documentation** first
2. **Search existing issues** and discussions
3. **Create a discussion** for questions
4. **Join our community** channels

## 🏆 Recognition

Contributors will be:

- **Listed in the README** acknowledgments
- **Credited in release notes**
- **Invited to be maintainers** (for significant contributions)

## 📋 Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] PR description is detailed
- [ ] Related issues are referenced

Thank you for contributing to Real-time Object Detection with YOLO! 🎯
