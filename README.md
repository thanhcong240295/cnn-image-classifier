# CNN Image Classifier - Rural vs Urban

Binary image classification model using Convolutional Neural Networks (CNN) to classify images as either Rural or Urban.

## Features

- **Data Augmentation**: RandomFlip (horizontal), RandomRotation, RandomZoom (reduced augmentation for better generalization)
- **Model Architecture**: Sequential CNN with Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, and Dropout layers
- **Class Balancing**: Automatic class weight balancing for imbalanced datasets
- **Model Checkpointing**: Saves best model during training based on validation loss
- **Early Stopping**: Prevents overfitting with configurable patience
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC with zero_division handling
- **Visualization**: Training history plots, confusion matrix, ROC curve, prediction distribution
- **Flexible Prediction**: Single image or batch folder prediction with CSV export
- **Model Persistence**: Save and load trained models (final and best checkpoint)

## Requirements

- Python 3.9.6
- TensorFlow 2.20.0
- Keras 3.10.0
- NumPy 2.0.2
- Matplotlib 3.9.4
- scikit-learn 1.6.1
- Pillow 11.3.0

## Installation

```bash
# Create virtual environment with Python 3.9.6
py -3.9 -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib scikit-learn pillow tensorflow keras
```

## Dataset Structure

```
du_lieu_goc/
├── rural/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── urban/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

## Configuration

Key parameters in `CNNImageClassifier` class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | (240, 240) | Input image dimensions |
| `EPOCHS` | 30 | Maximum training epochs |
| `BATCH_SIZE` | 16 | Training batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `DROPOUT_RATE` | 0.3 | Dropout rate for regularization |
| `TRAIN_SPLIT` | 0.80 | Training data split ratio |
| `VAL_SPLIT` | 0.20 | Validation data split ratio |
| `PREDICTION_THRESHOLD` | 0.5 | Classification threshold |
| `UNCERTAIN_CUTOFF` | 0.7 | Confidence threshold for uncertainty warning |
| `EARLY_STOPPING_PATIENCE` | 7 | Early stopping patience |

## Usage

### 1. Train Model

```bash
python cnn-image-classifier.py
```

This will:
- Load images from `du_lieu_goc/` folder
- Split data 80/20 (train/validation)
- Train the CNN model
- Save model to `model/cnn_image_classifier.keras`
- Generate visualizations in `result/` folder
- Display evaluation metrics

### 2. Single Image Prediction

```bash
# Train and predict
python cnn-image-classifier.py path/to/image.jpg

# Load existing model and predict
python cnn-image-classifier.py --load path/to/image.jpg
```

### 3. Batch Folder Prediction

```bash
# Train and predict folder
python cnn-image-classifier.py path/to/folder/

# Load existing model and predict folder
python cnn-image-classifier.py --load path/to/folder/ output.csv
```

## Model Architecture

```
Input (240x240x3)
├── Data Augmentation Layers
│   ├── RandomFlip (horizontal)
│   ├── RandomRotation (0.05)
│   └── RandomZoom (0.1)
├── Conv2D (32 filters, 3x3, padding='same')
├── BatchNormalization
├── MaxPooling2D (2x2) → 120x120
├── Conv2D (64 filters, 3x3, padding='same')
├── BatchNormalization
├── MaxPooling2D (2x2) → 60x60
├── Conv2D (128 filters, 3x3, padding='same')
├── BatchNormalization
├── MaxPooling2D (2x2) → 30x30
├── GlobalAveragePooling2D → 128 features
├── Dense (256 units, ReLU)
├── Dropout (0.3)
└── Dense (1 unit, Sigmoid)
```

**Key Architecture Improvements:**
- **GlobalAveragePooling2D** instead of Flatten reduces parameters and prevents overfitting
- **Reduced augmentation** (no vertical flip, brightness, contrast) for more stable training
- **Lower dropout** (0.3 vs 0.5) to retain more learned features
- **Class weight balancing** automatically handles imbalanced datasets

## Output Files

### Model
- `model/cnn_image_classifier.keras` - Final trained model
- `model/cnn_image_classifier_best.keras` - Best model checkpoint (based on val_loss)

### Visualizations (in `result/` folder)
- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Classification confusion matrix
- `roc_curve.png` - ROC curve with AUC score
- `prediction_distribution.png` - Distribution of prediction probabilities

### Predictions
- `result.csv` - Batch prediction results with paths, predictions, and probabilities

## Evaluation Metrics

The model outputs:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Example Output

```
Train set size: 160 (80%)
Validation set size: 40 (20%)

Test Loss: 0.3245
Test Accuracy: 0.8750

EVALUATION METRICS
============================================================
Accuracy:  0.8750
Precision: 0.8824
Recall:    0.8750
F1-Score:  0.8756
ROC-AUC:   0.9321
============================================================
```

## Troubleshooting

### TensorFlow Compatibility
- Use Python 3.9.6 for full TensorFlow 2.20.0 support
- Python 3.14+ is not yet supported by TensorFlow

### Out of Memory
- Reduce `BATCH_SIZE` (e.g., from 16 to 8)
- Reduce `IMG_SIZE` (e.g., from (240, 240) to (150, 150))

### Overfitting
- Increase `DROPOUT_RATE` (from 0.3 to 0.4 or 0.5)
- Reduce `EARLY_STOPPING_PATIENCE` (from 7 to 3 or 5)
- Add more training data or augmentation
- Enable more augmentation layers (vertical flip, brightness, contrast)

### Class Imbalance
- The model automatically uses `class_weight='balanced'`
- Consider collecting more samples from minority class
- Check confusion matrix to identify which class is problematic

### Low Accuracy
- Increase `EPOCHS` (from 30 to 50)
- Adjust `LEARNING_RATE` (try 0.0001 or 0.0005)
- Increase `EARLY_STOPPING_PATIENCE` to allow longer training
- Check data quality and ensure images are correctly labeled

## License

Educational project for image classification demonstration.

