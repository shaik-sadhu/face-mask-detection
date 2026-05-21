import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# Data path
DATA_PATH = 'data'

# Categories
CATEGORIES = ['with_mask', 'without_mask']

images = []
labels = []

print("Loading dataset...")

# Load images
for category in CATEGORIES:

    path = os.path.join(DATA_PATH, category)

    label = CATEGORIES.index(category)

    for img_name in os.listdir(path):

        img_path = os.path.join(path, img_name)

        try:
            img = cv2.imread(img_path)

            img = cv2.resize(img, (128, 128))

            images.append(img)

            labels.append(label)

        except:
            pass

print("Dataset Loaded Successfully!")

# Convert to numpy arrays
X = np.array(images) / 255.0

Y = to_categorical(labels)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Building CNN Model...")

# Build CNN Model
model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Started...")

# Train model
model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=10,
    batch_size=32
)

# Create model folder
os.makedirs('model', exist_ok=True)

# Save trained model
model.save('model/mask_detector.h5')

print("Model Saved Successfully!")