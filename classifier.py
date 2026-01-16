import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split


class CNNImageClassifier:
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
    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
    RESCALE_FACTOR = 1 / 255
    PREDICTION_THRESHOLD = 0.5
    RANDOM_STATE = 42
    UNCERTAIN_CUTOFF = 0.7
    
    def __init__(self, use_gpu=None, source=None, model_path=None):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.history = None
        self.device = None
        self.source = source
        self.class_names = None
        self.use_gpu = use_gpu if use_gpu is not None else True
        self.model_path = model_path
        if model_path:
            base, ext = os.path.splitext(model_path)
            self.model_best_path = f"{base}_best{ext}"
        else:
            self.model_best_path = None
        self.configure_device()
    
    def configure_device(self):
        gpus = tf.config.list_physical_devices('GPU')
        if self.use_gpu and gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus, 'GPU')
                self.device = 'GPU'
            except RuntimeError:
                self.device = 'CPU'
                tf.config.set_visible_devices([], 'GPU')
        else:
            tf.config.set_visible_devices([], 'GPU')
            self.device = 'CPU'
    
    def get_device_info(self):
        return {
            'device': self.device,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'tensorflow_version': tf.__version__,
            'cuda_support': tf.test.is_built_with_cuda()
        }

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
            filepath = self.model_path
        if filepath is None or self.model is None:
            return
        model_dir = os.path.dirname(filepath)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        try:
            self.model.save(filepath)
            if self.class_names:
                import json
                class_names_path = filepath.rsplit('.', 1)[0] + '_classes.json'
                with open(class_names_path, 'w') as f:
                    json.dump(self.class_names, f)
        except:
            pass
    
    def load_model(self, filepath=None):
        if filepath is None:
            filepath = self.model_best_path
        if filepath is None or not os.path.exists(filepath):
            return False
        try:
            self.model = load_model(filepath)
            import json
            class_names_path = filepath.rsplit('.', 1)[0] + '_classes.json'
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
            else:
                if self.source:
                    train_folder = os.path.join(os.path.dirname(__file__), self.source)
                    if os.path.exists(train_folder) and os.path.isdir(train_folder):
                        subdirs = sorted([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
                        if len(subdirs) == 2:
                            self.class_names = subdirs
            return True
        except:
            return False

    def summary(self):
        self.model.summary()

    def load_images_from_folder(self, folder_path, img_size=None):
        if img_size is None:
            img_size = self.IMG_SIZE
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        if len(subdirs) != 2:
            sys.exit(1)
        images = []
        labels = []
        class_labels = {}
        label_idx = 0
        for class_name in sorted(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_path):
                continue
            class_labels[class_name] = label_idx
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                if not img_file.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                    continue
                try:
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label_idx)
                except:
                    pass
            label_idx += 1
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
        return X_train, X_val, y_train, y_val

    def normalize_images(self):
        if self.X_train is None:
            return
        self.X_train = self.X_train.astype('float32') * self.RESCALE_FACTOR
        self.X_val = self.X_val.astype('float32') * self.RESCALE_FACTOR
        self.X_test = self.X_test.astype('float32') * self.RESCALE_FACTOR

    def train(self, epochs=None, batch_size=None):
        if epochs is None:
            epochs = self.EPOCHS
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        if self.X_train is None or self.y_train is None:
            return
        callbacks = []
        if self.USE_EARLY_STOPPING:
            if self.model_best_path:
                model_checkpoint = ModelCheckpoint(
                    filepath=self.model_best_path,
                    monitor=self.EARLY_STOPPING_MONITOR,
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(model_checkpoint)
            early_stopping = EarlyStopping(
                monitor=self.EARLY_STOPPING_MONITOR,
                patience=self.EARLY_STOPPING_PATIENCE,
                min_delta=self.EARLY_STOPPING_MIN_DELTA,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.EARLY_STOPPING_MONITOR,
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )

    def test(self):
        if self.X_test is None or self.y_test is None:
            return
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        loss = results[0]
        accuracy = results[1]
        return loss, accuracy
    
    def _preprocess_image(self, image_path):
        img = load_img(image_path, target_size=self.IMG_SIZE)
        img_array = img_to_array(img)
        img_normalized = img_array.astype('float32') * self.RESCALE_FACTOR
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    
    def _interpret_prediction(self, prediction):
        class_name = self.class_names[1] if prediction > self.PREDICTION_THRESHOLD else self.class_names[0]
        confidence = prediction if prediction > self.PREDICTION_THRESHOLD else 1 - prediction
        is_uncertain = max(prediction, 1 - prediction) < self.UNCERTAIN_CUTOFF
        return class_name, confidence, is_uncertain

    def predict_image(self, image_path):
        if self.model is None:
            return None
        if self.class_names is None:
            return None
        try:
            img_batch = self._preprocess_image(image_path)
            prediction = self.model.predict(img_batch, verbose=0)[0][0]
            class_name, confidence, is_uncertain = self._interpret_prediction(prediction)
            return {
                'image': image_path,
                'class': class_name,
                'confidence': confidence,
                'raw_prediction': prediction
            }
        except Exception as e:
            return None

    def predict_folder(self, folder_path, output_csv='result.csv'):
        if self.model is None:
            return None
        if self.class_names is None:
            return None
        results = []
        image_files = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)]
        if not image_files:
            return None
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
            except Exception as e:
                results.append({
                    'path': img_path,
                    'pred': 'ERROR',
                    'prob': 0.0
                })
        return results

    def run_pipeline(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        if epochs is None:
            epochs = self.EPOCHS
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        self.load_data(X_train, y_train, X_val, y_val)
        self.normalize_images()
        self.build_model()
        self.compile_model()
        self.summary()
        self.train(epochs=epochs, batch_size=batch_size)
        loss, accuracy = self.test()
        return loss, accuracy

    def main(self):
        if self.source is None:
            sys.exit(1)
        device_info = self.get_device_info()
        train_folder = os.path.join(os.path.dirname(__file__), self.source)
        if not os.path.exists(train_folder) or not os.path.isdir(train_folder):
            sys.exit(1)
        images, labels, class_labels = self.load_images_from_folder(train_folder)
        if len(images) == 0:
            return
        self.class_names = sorted(class_labels.keys())
        X_train, X_val, y_train, y_val = self.split_data_80_20(images, labels)
        y_train = y_train.astype('float32')
        y_val = y_val.astype('float32')
        self.run_pipeline(X_train, y_train, X_val, y_val)
        self.save_model()
