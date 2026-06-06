import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

# 1. Load Data
# MNIST dataset contains 60,000 training images and 10,000 testing images
# Images are 28x28 grayscale
print("Loading MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. Preprocess Data
# Reshape images to (28, 28, 1) for the CNN (height, width, channels)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
# This helps the model converge faster
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-hot encode labels (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 3. Build Neural Network Model
# Using a Convolutional Neural Network (CNN) which is effective for image recognition
model = models.Sequential()

# First Convolutional Layer: Extracts features like edges
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# MaxPooling: Reduces dimensionality and standardizes features
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer: Extracts more complex features
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer (Optional but good for accuracy)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten: Converts 3D feature maps to 1D feature vectors
model.add(layers.Flatten())

# Dense Layer: Fully connected layer for classification logic
model.add(layers.Dense(64, activation='relu'))

# Dropout: randomly sets input units to 0 to prevent overfitting
model.add(layers.Dropout(0.5))

# Output Layer: 10 neurons for 10 digits (0-9), using softmax for probability distribution
model.add(layers.Dense(10, activation='softmax'))

# 4. Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# 5. Train Model
# Train for 5 epochs (sufficient for MNIST to reach >98%)
print("Starting training...")
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# 6. Evaluate Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Save Model
model_save_path = "model.h5"
model.save(model_save_path)
print(f"Model saved to {os.path.abspath(model_save_path)}")
