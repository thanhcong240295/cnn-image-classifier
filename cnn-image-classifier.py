import os
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


class CNNImageClassifier:
    SOURCE = "du_lieu_goc"
    IMG_SIZE = (240, 240)
    IMG_CHANNELS = 3
    EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    USE_BATCH_NORM = True
    USE_AUGMENTATION = True
    USE_ROC_AUC = True
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_MIN_DELTA = 0.001
    TRAIN_SPLIT = 0.80
    VAL_SPLIT = 0.20
    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg')
    RESCALE_FACTOR = 1 / 255
    PREDICTION_THRESHOLD = 0.5
    MODEL_SAVE_PATH = 'model/cnn_image_classifier.keras'
    MODEL_BEST_SAVE_PATH = 'model/cnn_image_classifier_best.keras'
    RANDOM_STATE = 42
    UNCERTAIN_CUTOFF = 0.7
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.history = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], self.IMG_CHANNELS)))
        
        if self.USE_AUGMENTATION:
            self.model.add(RandomFlip("horizontal"))
            self.model.add(RandomRotation(0.05))
            self.model.add(RandomZoom(0.1))
        
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        if self.USE_BATCH_NORM:
            self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        if self.USE_BATCH_NORM:
            self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        if self.USE_BATCH_NORM:
            self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dropout(self.DROPOUT_RATE))
        self.model.add(Dense(units=1, activation='sigmoid'))

    def compile_model(self):
        optimizer = Adam(learning_rate=self.LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.MODEL_SAVE_PATH
        
        if self.model is None:
            print("Error: No model to save. Build and train the model first.")
            return
        
        model_dir = os.path.dirname(filepath)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        try:
            self.model.save(filepath)
            print(f"\nModel saved successfully to: {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath=None):
        if filepath is None:
            filepath = self.MODEL_BEST_SAVE_PATH
        
        if not os.path.exists(filepath):
            print(f"Error: Model file not found at {filepath}")
            return False
        
        try:
            self.model = load_model(filepath)
            print(f"\nModel loaded successfully from: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def summary(self):
        self.model.summary()

    def load_images_from_folder(self, folder_path, img_size=None):
        if img_size is None:
            img_size = self.IMG_SIZE
            
        images = []
        labels = []
        class_labels = {}
        label_idx = 0
        
        print(f"Loading images from {folder_path}...")
        
        for class_name in sorted(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            class_labels[class_name] = label_idx
            print(f"  Loading class '{class_name}' (label={label_idx})...")
            
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                
                if not img_file.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                    continue
                
                try:
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"    Error loading {img_file}: {e}")
            
            label_idx += 1
        
        print(f"Total images loaded: {len(images)}")
        return np.array(images), np.array(labels), class_labels

    def load_data(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_val
        self.y_test = y_val

    def split_data_80_20(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=self.RANDOM_STATE
        )
        
        print(f"Train set size: {len(X_train)} (80%)")
        print(f"Validation set size: {len(X_val)} (20%)")
        
        return X_train, X_val, y_train, y_val

    def normalize_images(self):
        if self.X_train is None:
            print("Error: Load data first using load_data()")
            return
        
        print("Normalizing images...")
        self.X_train = self.X_train.astype('float32') * self.RESCALE_FACTOR
        self.X_val = self.X_val.astype('float32') * self.RESCALE_FACTOR
        self.X_test = self.X_test.astype('float32') * self.RESCALE_FACTOR
        print("Images normalized successfully")

    def train(self, epochs=None, batch_size=None):
        if epochs is None:
            epochs = self.EPOCHS
        if batch_size is None:
            batch_size = self.BATCH_SIZE
            
        if self.X_train is None or self.y_train is None:
            print("Error: Training data not loaded. Call load_data() first.")
            return
        
        print(f"Training model for {epochs} epochs...")
        
        callbacks = []
        if self.USE_EARLY_STOPPING:
            model_checkpoint = ModelCheckpoint(
                filepath=self.MODEL_BEST_SAVE_PATH,
                monitor=self.EARLY_STOPPING_MONITOR,
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)
            print(f"Using ModelCheckpoint to save best model to {self.MODEL_BEST_SAVE_PATH}")

            early_stopping = EarlyStopping(
                monitor=self.EARLY_STOPPING_MONITOR,
                patience=self.EARLY_STOPPING_PATIENCE,
                min_delta=self.EARLY_STOPPING_MIN_DELTA,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            print(f"Using EarlyStopping (monitor={self.EARLY_STOPPING_MONITOR}, patience={self.EARLY_STOPPING_PATIENCE})")
        
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.EARLY_STOPPING_MONITOR,
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
            print("Using ReduceLROnPlateau to reduce learning rate on plateau")

        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        print("Training completed")

    def test(self):
        if self.X_test is None or self.y_test is None:
            print("Error: Test data not loaded. Call load_data() first.")
            return
        
        print("Testing model...")
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        loss = results[0]
        accuracy = results[1]
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        return loss, accuracy
    
    def _preprocess_image(self, image_path):
        img = load_img(image_path, target_size=self.IMG_SIZE)
        img_array = img_to_array(img)
        img_normalized = img_array.astype('float32') * self.RESCALE_FACTOR
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    
    def _interpret_prediction(self, prediction):
        class_name = "Urban" if prediction > self.PREDICTION_THRESHOLD else "Rural"
        confidence = prediction if prediction > self.PREDICTION_THRESHOLD else 1 - prediction
        is_uncertain = max(prediction, 1 - prediction) < self.UNCERTAIN_CUTOFF
        return class_name, confidence, is_uncertain

    def predict_image(self, image_path):
        if self.model is None:
            print("Error: Model not initialized. Train or load a model first.")
            return None
        
        try:
            print(f"Loading image: {image_path}")
            img_batch = self._preprocess_image(image_path)
            
            prediction = self.model.predict(img_batch, verbose=0)[0][0]
            class_name, confidence, is_uncertain = self._interpret_prediction(prediction)
            
            if is_uncertain:
                print("Warning: The model is uncertain about this prediction.")
            
            print(f"\n{'='*50}")
            print(f"Prediction Result:")
            print(f"{'='*50}")
            print(f"Image: {image_path}")
            print(f"Classification: {class_name}")
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Raw prediction: {prediction:.4f}")
            print(f"{'='*50}\n")
            
            return {
                'image': image_path,
                'class': class_name,
                'confidence': confidence,
                'raw_prediction': prediction
            }
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None

    def predict_folder(self, folder_path, output_csv='result.csv'):
        if self.model is None:
            print("Error: Model not initialized. Train or load a model first.")
            return None
        
        results = []
        print(f"\nPredicting images in folder: {folder_path}")
        print("="*60)
        
        image_files = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)]
        
        if not image_files:
            print("No images found in folder!")
            return None
        
        print(f"Found {len(image_files)} images to predict...\n")
        
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(folder_path, img_file)
            
            try:
                img_batch = self._preprocess_image(img_path)
                prediction = self.model.predict(img_batch, verbose=0)[0][0]
                class_name, probability, _ = self._interpret_prediction(prediction)
                
                results.append({
                    'path': img_path,
                    'pred': class_name,
                    'prob': probability
                })
                
                print(f"[{idx}/{len(image_files)}] {img_file}: {class_name} ({probability*100:.2f}%)")
                
            except Exception as e:
                print(f"[{idx}/{len(image_files)}] Error processing {img_file}: {e}")
                results.append({
                    'path': img_path,
                    'pred': 'ERROR',
                    'prob': 0.0
                })
        
        self._save_predictions_to_csv(results, output_csv)
        return results
    
    def _save_predictions_to_csv(self, results, output_csv):
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['path', 'pred', 'prob']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n{'='*60}")
            print(f"Results saved to: {output_csv}")
            print(f"Total images processed: {len(results)}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def plot_training_history(self, save_path=None):
        if self.history is None:
            print("Error: No training history found. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        y_pred_binary = (y_pred > self.PREDICTION_THRESHOLD).astype(int).flatten()
        y_true_binary = y_true.astype(int).flatten()
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rural', 'Urban'])
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('Confusion Matrix - Rural vs Urban', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to: {save_path}")
        
        plt.close()

    def plot_prediction_distribution(self, predictions, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(predictions, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=self.PREDICTION_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Decision Boundary ({self.PREDICTION_THRESHOLD})')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Model Predictions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction distribution plot saved to: {save_path}")
        
        plt.close()

    def plot_roc_curve(self, y_true, y_pred, save_path=None):
        y_true_binary = y_true.astype(int).flatten()
        y_pred_flat = y_pred.flatten()
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_flat)
        roc_auc = roc_auc_score(y_true_binary, y_pred_flat)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve plot saved to: {save_path}")
        
        plt.close()
        return roc_auc

    def generate_report(self, y_true, y_pred):
        y_pred_binary = (y_pred > self.PREDICTION_THRESHOLD).astype(int).flatten()
        y_true_binary = y_true.astype(int).flatten()
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        if self.USE_ROC_AUC:
            y_pred_flat = y_pred.flatten()
            roc_auc = roc_auc_score(y_true_binary, y_pred_flat)
            print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print("="*60)
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true_binary, y_pred_binary, 
                                   target_names=['Rural', 'Urban'],
                                   digits=4,
                                   zero_division=0))
        print("="*60 + "\n")

    def run_pipeline(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        if epochs is None:
            epochs = self.EPOCHS
        if batch_size is None:
            batch_size = self.BATCH_SIZE

        print("Starting CNN Image Classifier Pipeline")

        print("\n[Step 1/6] Loading data...")
        self.load_data(X_train, y_train, X_val, y_val)
        
        print("\n[Step 2/6] Normalizing images...")
        self.normalize_images()
        
        print("\n[Step 3/6] Building model...")
        self.build_model()
        
        print("\n[Step 4/6] Compiling model...")
        self.compile_model()
        
        print("\n[Step 5/6] Model summary:")
        self.summary()
        
        print("\n[Step 6/6] Training model...")
        self.train(epochs=epochs, batch_size=batch_size)
        
        print("\nTesting model...")
        loss, accuracy = self.test()

        return loss, accuracy

    def main(self):
        train_folder = os.path.join(os.path.dirname(__file__), self.SOURCE)

        images, labels, class_labels = self.load_images_from_folder(train_folder)
        
        if len(images) == 0:
            print("Error: No images found in train folder!")
            return
        
        print(f"Class labels: {class_labels}")
        
        X_train, X_val, y_train, y_val = self.split_data_80_20(images, labels)
        
        y_train = y_train.astype('float32')
        y_val = y_val.astype('float32')

        loss, accuracy = self.run_pipeline(X_train, y_train, X_val, y_val)
        
        print(f"\nFinal Results: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        self.save_model()
        
        try:
            print("\nGenerating visualizations...")
            result_dir = os.path.join(os.path.dirname(__file__), 'result')
            os.makedirs(result_dir, exist_ok=True)
            
            self.plot_training_history(save_path=os.path.join(result_dir, 'training_history.png'))
            
            test_predictions = self.model.predict(self.X_test, verbose=0)
            self.plot_confusion_matrix(self.y_test, test_predictions, save_path=os.path.join(result_dir, 'confusion_matrix.png'))
            self.plot_prediction_distribution(test_predictions, save_path=os.path.join(result_dir, 'prediction_distribution.png'))
            
            if self.USE_ROC_AUC:
                self.plot_roc_curve(self.y_test, test_predictions, save_path=os.path.join(result_dir, 'roc_curve.png'))
            
            self.generate_report(self.y_test, test_predictions)
        except Exception as e:
            print(f"Error generating visualizations: {e}")


if __name__ == '__main__':
    classifier = CNNImageClassifier()
    
    load_model_flag = '--load' in sys.argv
    if load_model_flag:
        sys.argv.remove('--load')
        if classifier.load_model():
            print("Using loaded model for predictions\n")
        else:
            print("Failed to load model. Will train from scratch.\n")
            load_model_flag = False
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if os.path.isdir(arg):
            print("Folder prediction mode")
            output_csv = sys.argv[2] if len(sys.argv) > 2 else 'result.csv'
            
            if not load_model_flag:
                classifier.main()
            
            classifier.predict_folder(arg, output_csv)
        else:
            print("Single image prediction mode")
            
            if not load_model_flag:
                classifier.main()
            
            classifier.predict_image(arg)
    else:
        classifier.main()