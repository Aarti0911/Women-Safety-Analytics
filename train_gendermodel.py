import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

#  Dataset path
DATASET_DIR = r'C:\Users\HP\women-safety-env\models\utkface_aligned_cropped'

# Load image paths recursively
print(" Loading dataset...")

image_paths = []
for root, _, filenames in os.walk(DATASET_DIR):
    for filename in filenames:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, filename))

#  Sanity check
print(f" Found {len(image_paths)} images.")
if len(image_paths) > 0:
    print(" Example:", image_paths[0])
else:
    raise ValueError(" No images were loaded. Check your DATASET_DIR or image formats.")

#  Prepare data
X, y = [], []

for path in image_paths:
    try:
        filename = os.path.basename(path)
        parts = filename.split("_")
        gender = int(parts[1])

        if gender not in [0, 1]:
            continue  #  Skip anything that's not 0 or 1

        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(gender)

    except Exception as e:
        print(f"Skipping {path} due to error: {e}")


X = np.array(X, dtype="float32") / 255.0
y = to_categorical(y, num_classes=2)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: Male, Female
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Save best model
checkpoint = ModelCheckpoint("gender_model_best.h5", monitor='val_accuracy', save_best_only=True)

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, callbacks=[checkpoint])
