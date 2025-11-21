# CI/CD Pipeline Documentation

## Overview

This GitHub Actions CI/CD pipeline automatically runs tests, checks code quality, and validates the simulation on every push and pull request.

## Pipeline Stages

### 1. Test Job
**Runs on**: Ubuntu Latest with Python 3.10 and 3.11

**Steps**:
- Installs dependencies from `requirements.txt`
- Runs `flake8` linting for critical errors (syntax, undefined names)
- Runs `mypy` type checking (continues on error)
- Executes pytest with coverage reporting
- Uploads coverage to Codecov
- **Pass Criteria**: 50% code coverage minimum, all critical tests pass

### 2. Simulation Test Job
**Runs on**: Ubuntu Latest with Python 3.11
**Depends on**: Test job must pass first

**Steps**:
- Runs a 50-step, 50-agent simulation
- Verifies output files are created (trades.csv, agent_metrics.csv)
- **Pass Criteria**: Simulation completes without errors

### 3. Security Job
**Runs on**: Ubuntu Latest with Python 3.11

**Steps**:
- Runs `safety` to check for known vulnerabilities in dependencies
- Runs `bandit` for security issues in source code
- Uploads security report as artifact
- **Pass Criteria**: Continues on error (informational only)

## Local Testing

### Run Full Test Suite
```bash
source venv/bin/activate
export PYTHONPATH=$(pwd)
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=50 -v
```

### Run Linting
```bash
# Critical errors only
flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics

# Full style check
flake8 src --count --exit-zero --max-complexity=10 --statistics
```

### Run Type Checking
```bash
mypy src --ignore-missing-imports --no-strict-optional
```

### Run Security Checks
```bash
# Check dependencies
safety check --file requirements.txt

# Check source code
bandit -r src
```

### Run Simulation Test
```bash
python src/main.py --steps 50 --agents 50
ls -lh output/
```

## Configuration Files

- **`.github/workflows/ci.yml`**: GitHub Actions workflow definition
- **`.flake8`**: Flake8 linting configuration
- **`mypy.ini`**: MyPy type checking configuration
- **`pytest.ini`**: Pytest configuration with coverage settings

## Current Test Status

- **Total Tests**: 61
- **Passing**: 59 (96.7%)
- **Code Coverage**: 53% (exceeds 50% requirement)
- **Known Issues**: 2 non-critical failures (probabilistic test, FOK edge case)

## Test Markers

Use markers to run specific test categories:

```bash
pytest -m unit           # Run unit tests only
pytest -m integration    # Run integration tests only
pytest -m orderbook      # Run orderbook tests only
pytest -m agents         # Run agent tests only
pytest -m models         # Run model tests only
```

## Coverage Reporting

After running tests, view detailed coverage:

```bash
# Terminal output
pytest --cov=src --cov-report=term-missing

# HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Troubleshooting

### Import Errors
Ensure `PYTHONPATH` is set:
```bash
export PYTHONPATH=$(pwd)
```

### Coverage Below Threshold
Check which files have low coverage:
```bash
pytest --cov=src --cov-report=term-missing | grep -E "^src/"
```

### Flake8 Errors
View all style issues:
```bash
flake8 src --show-source --statistics
```

## Next Steps

1. Fix remaining 2 test failures for 100% pass rate
2. Increase coverage to 70% (add integration tests)
3. Add pre-commit hooks for local validation
4. Set up branch protection rules requiring CI pass
5. Configure Codecov for PR comments with coverage diffs
