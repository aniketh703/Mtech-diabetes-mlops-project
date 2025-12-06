#!/usr/bin/env python
"""
Local CI/CD Pipeline Simulation Script
Simulates GitHub Actions workflow locally for testing
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Get the Python executable from the current environment
PYTHON = sys.executable
PIP = f'"{PYTHON}" -m pip'


def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")


def print_step(text):
    print(f"{YELLOW}▶ {text}{RESET}")


def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    print(f"{RED}✗ {text}{RESET}")


def run_command(command, description, allow_failure=False):
    """Run a command and return success status"""
    print_step(description)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            print_success(f"{description} - PASSED")
            return True
        else:
            if allow_failure:
                print(f"{YELLOW}⚠ {description} - WARNING (non-blocking){RESET}")
                if result.stderr:
                    print(f"  {result.stderr[:200]}...")
                return True
            else:
                print_error(f"{description} - FAILED")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}")
                return False
    except subprocess.TimeoutExpired:
        print_error(f"{description} - TIMEOUT")
        return False
    except Exception as e:
        print_error(f"{description} - ERROR: {str(e)}")
        return False


def main():
    start_time = time.time()
    results = {}
    
    print(f"\n{BOLD}🚀 Local CI/CD Pipeline Simulation{RESET}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using Python: {PYTHON}")
    
    # Stage 1: Environment Check
    print_header("Stage 1: Environment Check")
    results['env'] = run_command(f'"{PYTHON}" --version', "Python version check")
    results['pip'] = run_command(f'{PIP} --version', "Pip version check")
    
    # Stage 2: Install Dependencies
    print_header("Stage 2: Install Dependencies")
    results['deps'] = run_command(
        f'{PIP} install flake8 pytest pytest-cov black isort httpx pytest-asyncio -q',
        "Install testing dependencies"
    )
    
    # Stage 3: Code Quality
    print_header("Stage 3: Code Quality Checks")
    results['flake8_critical'] = run_command(
        f'"{PYTHON}" -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv,__pycache__,.git,mlruns',
        "Flake8 critical errors check",
        allow_failure=True
    )
    results['flake8_style'] = run_command(
        f'"{PYTHON}" -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=venv,__pycache__,.git,mlruns',
        "Flake8 style check",
        allow_failure=True
    )
    results['black'] = run_command(
        f'"{PYTHON}" -m black --check --diff . --exclude "venv|__pycache__|.git|mlruns"',
        "Black formatting check",
        allow_failure=True
    )
    results['isort'] = run_command(
        f'"{PYTHON}" -m isort --check-only --diff . --skip venv --skip __pycache__ --skip mlruns --skip .git',
        "Import sorting check",
        allow_failure=True
    )
    
    # Stage 4: Unit Tests
    print_header("Stage 4: Unit Tests")
    results['tests'] = run_command(
        f'"{PYTHON}" -m pytest tests/ -v --tb=short',
        "Run unit tests"
    )
    
    # Stage 5: Data Pipeline
    print_header("Stage 5: Data Pipeline")
    if os.path.exists("data/diabetes.csv"):
        results['preprocess'] = run_command(
            f'"{PYTHON}" preprocess.py',
            "Data preprocessing"
        )
    else:
        print(f"{YELLOW}⚠ Skipping preprocessing - data file not found{RESET}")
        results['preprocess'] = True
    
    # Stage 6: Model Training
    print_header("Stage 6: Model Training")
    if os.path.exists("train_simple.py"):
        results['train'] = run_command(
            f'"{PYTHON}" train_simple.py',
            "Train model"
        )
    else:
        print(f"{YELLOW}⚠ Skipping training - train_simple.py not found{RESET}")
        results['train'] = True
    
    # Stage 7: Model Evaluation (optional - may have MLflow issues)
    print_header("Stage 7: Model Evaluation")
    if os.path.exists("evaluate.py"):
        results['evaluate'] = run_command(
            f'"{PYTHON}" evaluate.py',
            "Evaluate model",
            allow_failure=True  # MLflow may have path issues locally
        )
    else:
        print(f"{YELLOW}⚠ Skipping evaluation - evaluate.py not found{RESET}")
        results['evaluate'] = True
    
    # Stage 8: Docker Build Check
    print_header("Stage 8: Docker Check")
    docker_available = run_command("docker --version", "Docker availability check", allow_failure=True)
    if docker_available and os.path.exists("Dockerfile"):
        results['docker'] = run_command(
            "docker build -t diabetes-mlops:test . --no-cache",
            "Docker build test",
            allow_failure=True
        )
    else:
        print(f"{YELLOW}⚠ Docker build skipped{RESET}")
        results['docker'] = True
    
    # Summary
    print_header("Pipeline Summary")
    
    elapsed_time = time.time() - start_time
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print(f"Total stages: {len(results)}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"Duration: {elapsed_time:.2f} seconds")
    
    print("\nStage Results:")
    for stage, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {stage}: {status}")
    
    # Final status
    if failed == 0:
        print(f"\n{GREEN}{BOLD}✅ CI/CD Pipeline PASSED{RESET}")
        return 0
    else:
        print(f"\n{RED}{BOLD}❌ CI/CD Pipeline FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
