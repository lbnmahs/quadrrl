# Contributing to Quadrrl

Thank you for your interest in contributing to Quadrrl! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment for all contributors

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Include:
  - Clear description of the issue
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior
  - Environment details (OS, Isaac Lab version, Python version)

### Submitting Changes

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow existing code style (PEP 8 for Python)
   - Add docstrings for new functions/classes
   - Update relevant documentation

3. **Test your changes**
   ```bash
   # Run pre-commit hooks
   pre-commit run --all-files
   
   # Test that environments still register
   python scripts/list_envs.py
   ```

4. **Commit your changes**
   - Write clear, descriptive commit messages
   - Reference issue numbers if applicable

5. **Push and create a Pull Request**
   - Provide a clear description of changes
   - Link to any related issues
   - Request review from maintainers

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Keep functions focused and modular
- Add docstrings (Google or NumPy style)

### Adding New Tasks

1. Create task directory following existing structure
2. Implement environment class (`*_env.py`)
3. Create configuration (`*_env_cfg.py`)
4. Add agent configs for desired RL frameworks
5. Update `scripts/list_envs.py` if needed
6. Add documentation in `docs/TRAINING.md`

### Adding New Robots

1. Create robot file in `source/quadrrl/quadrrl/robots/`
2. Define robot configuration class
3. Register in appropriate task configs
4. Update documentation

### Testing

- Test new environments with `scripts/list_envs.py`
- Verify training scripts work with new tasks
- Test on both Linux and Windows if possible

## Academic Contributions

If you use Quadrrl in academic work, please cite:

```bibtex
@software{quadrrl2024,
  title={Quadrrl: The Benchmarking Suite for Quadruped Robot Reinforcement Learning},
  author={Mahihu, Laban Njoroge},
  year={2024},
  url={https://github.com/lbnmahs/quadrrl}
}
```

## License

By contributing, you agree that your contributions will be licensed under the BSD-3-Clause License.

## Questions?

Open an issue or contact the maintainers.

