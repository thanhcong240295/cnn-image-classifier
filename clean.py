import os
import shutil


def clean_project():
    """Remove generated files and directories"""
    base_dir = os.path.dirname(__file__)

    items_to_clean = [
        "model",
        "result",
        "__pycache__",
        "output.csv",
        "result.csv",
        ".pytest_cache",
        "*.pyc",
        "*.pyo",
        "*.log",
    ]

    for item in items_to_clean:
        path = os.path.join(base_dir, item)

        if "*" in item:
            import glob

            for file in glob.glob(path):
                try:
                    os.remove(file)
                    print(f"Removed: {file}")
                except:
                    pass
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            except:
                pass
        elif os.path.isfile(path):
            try:
                os.remove(path)
                print(f"Removed file: {path}")
            except:
                pass

    print("\nCleanup complete!")


if __name__ == "__main__":
    confirmation = input("This will remove all generated files (models, results, cache). Continue? (y/n): ")
    if confirmation.lower() == "y":
        clean_project()
    else:
        print("Cleanup cancelled.")
