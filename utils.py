import csv
import os
import sys


def validate_arguments():
    use_gpu = True
    source_dir = None
    model_path = None
    if "--cpu" in sys.argv:
        use_gpu = False
        sys.argv.remove("--cpu")
    elif "--gpu" in sys.argv:
        use_gpu = True
        sys.argv.remove("--gpu")
    if "--source" in sys.argv:
        source_idx = sys.argv.index("--source")
        if source_idx + 1 < len(sys.argv):
            source_dir = sys.argv[source_idx + 1]
            if not os.path.exists(source_dir):
                sys.exit(1)
            if not os.path.isdir(source_dir):
                sys.exit(1)
            sys.argv.pop(source_idx)
            sys.argv.pop(source_idx)
        else:
            sys.exit(1)
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            model_path = sys.argv[model_idx + 1]
            if not model_path.endswith((".keras", ".h5")):
                sys.exit(1)
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            sys.argv.pop(model_idx)
            sys.argv.pop(model_idx)
        else:
            sys.exit(1)
    return use_gpu, source_dir, model_path


def save_predictions_to_csv(results, output_csv):
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["path", "pred", "prob"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    except:
        pass
