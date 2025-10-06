"""Quick 2-epoch smoke-training script for CIFAR-10 used to collect metrics for the notebook analysis.
Saves a quick model to best_cnn_cifar10_quick.h5 and prints training/validation/test metrics.
"""
import os
import sys
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import to_categorical
except ModuleNotFoundError as e:
    print("ERROR: TensorFlow is not installed in this environment.")
    print("Install it with: pip install tensorflow-cpu")
    sys.exit(2)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print('TensorFlow version:', tf.__version__)

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Use a smaller subset for a quick smoke test
n_train = 5000
n_test = 1000
x_train = x_train[:n_train].astype('float32') / 255.0
x_test = x_test[:n_test].astype('float32') / 255.0
y_train = to_categorical(y_train[:n_train], 10)
y_test = to_categorical(y_test[:n_test], 10)

print('Using train samples:', x_train.shape[0], 'test samples:', x_test.shape[0])

# Build a small CNN (same architecture as notebook)
def build_cnn(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model = build_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train for 2 epochs (quick)
history = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.1, verbose=2)

# Save quick model
OUT = 'best_cnn_cifar10_quick.h5'
model.save(OUT)
print(f"Saved quick model to {OUT}")

# Evaluate on test subset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Quick test loss: {test_loss:.4f}")
print(f"Quick test accuracy: {test_acc:.4f}")

# Print final history metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print('Final train accuracy:', final_train_acc)
print('Final val accuracy:', final_val_acc)
print('Final train loss:', final_train_loss)
print('Final val loss:', final_val_loss)
