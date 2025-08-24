# GitHub Actions CI/CD Pipeline

This directory contains the GitHub Actions workflow for automated testing, building, and deployment of the StruktX AI package.

## Workflow Overview

The CI/CD pipeline consists of the following jobs:

### 1. Test Job
- **Trigger**: Push to main/develop, Pull Requests, Releases
- **Purpose**: Run comprehensive tests across multiple Python versions
- **Actions**:
  - Runs on Python 3.8, 3.9, 3.10, 3.11, 3.12
  - Installs dependencies using `uv`
  - Runs linting (black, isort, flake8)
  - Runs type checking (mypy)
  - Runs tests with pytest and coverage
  - Uploads coverage to Codecov

### 2. Build Job
- **Trigger**: Only on published releases
- **Purpose**: Build the package for distribution
- **Actions**:
  - Builds source distribution and wheel
  - Uploads build artifacts

### 3. Deploy Job
- **Trigger**: Only on published releases
- **Purpose**: Deploy to PyPI and optionally Test PyPI
- **Actions**:
  - Downloads build artifacts
  - Publishes to PyPI (production)
  - Publishes to Test PyPI (testing) - only if TEST_PYPI_API_TOKEN is provided

### 4. Create Release Job
- **Trigger**: Only on published releases
- **Purpose**: Update release notes with changelog
- **Actions**:
  - Generates changelog from git commits
  - Updates GitHub release with formatted notes

## Required Secrets

To enable deployment, you need to set up the following secrets in your GitHub repository:

1. **PYPI_API_TOKEN**: Your PyPI API token for production deployment (Required)
2. **TEST_PYPI_API_TOKEN**: Your Test PyPI API token for testing deployment (Optional)

### Setting up PyPI Tokens

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create an API token with "Entire account" scope
3. Add it as `PYPI_API_TOKEN` in your GitHub repository secrets

4. Go to [Test PyPI Account Settings](https://test.pypi.org/manage/account/)
5. Create an API token with "Entire account" scope
6. Add it as `TEST_PYPI_API_TOKEN` in your GitHub repository secrets

## How to Release

1. Update the version in `pyproject.toml`
2. Commit and push your changes
3. Create a new release on GitHub with the same version tag
4. The workflow will automatically:
   - Run all tests
   - Build the package
   - Deploy to PyPI and Test PyPI
   - Update the release notes

## Package Installation

After deployment, users can install the package with:

```bash
# Install with all optional dependencies
pip install struktx-ai[all]

# Install with specific optional dependencies
pip install struktx-ai[llm,vector]

# Install basic version only
pip install struktx-ai
```

## CLI Usage

After installation, users can use the CLI:

```bash
# Interactive mode
struktx-ai --interactive

# Single message
struktx-ai --message "Hello, how are you?"

# With custom config
struktx-ai --config config.yaml --message "Hello"
```
