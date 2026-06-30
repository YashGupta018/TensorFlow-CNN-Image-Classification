import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# ✅ Step 1: Setup Paths
BASE_PATH = r"F:\ABB Projects\TensorFlow-CNN-Image-Classification\data\mvtec_anomaly_detection-batches-py-target\cable"
TRAIN_PATH = os.path.join(BASE_PATH, "train", "good")
TEST_PATH = os.path.join(BASE_PATH, "test")
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# ✅ Step 2: Load Images
def load_images_from_folder(folder, label, img_size):
    images, labels = [], []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB").resize(img_size)
            images.append(np.array(img) / 255.0)
            labels.append(label)
    return images, labels

# Load good (0) and defective (1) images
print("✅ Loading training images...")
x_train, y_train = load_images_from_folder(TRAIN_PATH, 0, IMG_SIZE)

print("✅ Loading test images...")
x_test, y_test = [], []

# Load good test images
good_imgs, good_labels = load_images_from_folder(
    os.path.join(TEST_PATH, "good"), 0, IMG_SIZE)
x_test.extend(good_imgs)
y_test.extend(good_labels)

# Load all defective test images
defect_folders = [f for f in os.listdir(TEST_PATH) if f != "good"]
for folder in defect_folders:
    imgs, labels = load_images_from_folder(
        os.path.join(TEST_PATH, folder), 1, IMG_SIZE)
    x_test.extend(imgs)
    y_test.extend(labels)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(f"✅ Training samples: {len(x_train)}")
print(f"✅ Test samples: {len(x_test)}")

# ✅ Step 3: Build CNN Model
model = models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    # First Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Second Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Third Convolution Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary: good vs defective
])

# ✅ Step 4: Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ✅ Step 5: Train Model
print("\n✅ Training started...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test)
)
print("✅ Training complete!")

# ✅ Step 6: Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# ✅ Step 7: Save Model
model.save('abb_cable_defect_model.keras')
print("✅ Model saved as abb_cable_defect_model.keras")

# ✅ Step 8: Plot Training Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('ABB Cable Defect CNN - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('ABB Cable Defect CNN - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('mvtec_training_results.png')
plt.show()

# ✅ Step 9: Visualize Predictions
print("\n✅ Generating predictions...")
predictions = model.predict(x_test)
class_names = ['Good', 'Defective']

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    predicted_label = class_names[int(predictions[i] > 0.5)]
    actual_label = class_names[int(y_test[i])]
    color = 'green' if predicted_label == actual_label else 'red'
    plt.title(f"Pred: {predicted_label}\nActual: {actual_label}", color=color)
    plt.axis('off')

plt.suptitle('ABB Cable Defect Detection - Predictions', fontsize=14)
plt.tight_layout()
plt.savefig('mvtec_predictions.png')
plt.show()
print("✅ Predictions saved as mvtec_predictions.png")