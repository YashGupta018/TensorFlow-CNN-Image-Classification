import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# ✅ Step 1: Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# ✅ Step 2: Normalize pixel values to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# ABB-relevant class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("✅ Dataset loaded successfully!")
print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# ✅ Step 3: Build CNN Model
model = models.Sequential([
    # First Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# ✅ Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# ✅ Step 5: Train the Model
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=64,
                    validation_data=(x_test, y_test))

print("✅ Model training complete!")

# ✅ Step 6: Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# ✅ Step 7: Save the Model
model.save('abb_cnn_model.keras')
print("✅ Model saved as abb_cnn_model.keras")

# ✅ Step 8: Plot Training Results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('ABB CNN - Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('ABB CNN - Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
print("✅ Training results saved as training_results.png")