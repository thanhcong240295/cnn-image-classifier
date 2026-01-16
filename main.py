import os
import sys

from classifier import CNNImageClassifier
from utils import save_predictions_to_csv, validate_arguments
from visualization import (
    generate_report,
    plot_confusion_matrix,
    plot_prediction_distribution,
    plot_roc_curve,
    plot_training_history,
)


def main():
    use_gpu, source_dir, model_path = validate_arguments()
    classifier = CNNImageClassifier(use_gpu=use_gpu, source=source_dir, model_path=model_path)
    load_model_flag = "--load" in sys.argv
    if load_model_flag:
        sys.argv.remove("--load")
        if not classifier.load_model():
            load_model_flag = False

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if not os.path.exists(arg):
            sys.exit(1)
        if os.path.isdir(arg):
            output_csv = sys.argv[2] if len(sys.argv) > 2 else "result.csv"
            if not load_model_flag:
                classifier.main()
            results = classifier.predict_folder(arg, output_csv)
            if results:
                save_predictions_to_csv(results, output_csv)
            else:
                print("Prediction failed. Ensure model and class names are loaded correctly.")
                sys.exit(1)
        else:
            if not load_model_flag:
                classifier.main()
            classifier.predict_image(arg)
    else:
        classifier.main()
        try:
            result_dir = os.path.join(os.path.dirname(__file__), "result")
            os.makedirs(result_dir, exist_ok=True)
            plot_training_history(classifier.history, save_path=os.path.join(result_dir, "training_history.png"))
            test_predictions = classifier.model.predict(classifier.X_test, verbose=0)
            plot_confusion_matrix(
                classifier.y_test,
                test_predictions,
                classifier.class_names,
                classifier.PREDICTION_THRESHOLD,
                save_path=os.path.join(result_dir, "confusion_matrix.png"),
            )
            plot_prediction_distribution(
                test_predictions,
                classifier.PREDICTION_THRESHOLD,
                save_path=os.path.join(result_dir, "prediction_distribution.png"),
            )
            if classifier.USE_ROC_AUC:
                plot_roc_curve(classifier.y_test, test_predictions, save_path=os.path.join(result_dir, "roc_curve.png"))
            generate_report(
                classifier.y_test,
                test_predictions,
                classifier.class_names,
                classifier.PREDICTION_THRESHOLD,
                classifier.USE_ROC_AUC,
            )
        except:
            pass


if __name__ == "__main__":
    main()
