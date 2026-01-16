#!/usr/bin/env python
"""Code quality check script"""
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def main():
    """Run all code quality checks"""
    files = "classifier.py visualization.py utils.py main.py clean.py"

    checks = [
        (f"python -m black --check {files}", "Black (code formatter check)"),
        (f"python -m isort --check-only {files}", "isort (import sorting check)"),
        (f"python -m flake8 {files}", "flake8 (linting)"),
        (f"python -m pylint {files}", "pylint (static analysis)"),
    ]

    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("[OK] All checks passed!")
    else:
        print("[FAIL] Some checks failed. Run the following to fix:")
        print("  python format_code.py")
    print(f"{'='*60}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
