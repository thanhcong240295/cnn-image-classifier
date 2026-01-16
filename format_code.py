#!/usr/bin/env python
"""Auto-format code with black and isort"""
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\nRunning {description}...")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    """Format all Python files"""
    files = "classifier.py visualization.py utils.py main.py clean.py check_code.py format_code.py"

    commands = [
        (f"python -m isort {files}", "isort (sorting imports)"),
        (f"python -m black {files}", "black (formatting code)"),
    ]

    all_passed = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("[OK] Code formatted successfully!")
    else:
        print("[FAIL] Some formatting failed.")
    print(f"{'='*60}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
