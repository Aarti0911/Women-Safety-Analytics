import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ============================
# Function to Load UTKFace Data
# ============================
def load_utkface(dataset_path=r'C:\Users\HP\women-safety-env\UTKFace', target_size=(224, 224)):
    X = []
    y = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            try:
                # Format: [age]_[gender]_[race]_[date].jpg
                gender = int(filename.split("_")[1])  # 0 = Male, 1 = Female
                img_path = os.path.join(dataset_path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, target_size)
                X.append(img)
                y.append(gender)
            except Exception as e:
                print(f"Skipping file {filename}: {e}")
                continue

    X = np.array(X, dtype="float32") / 255.0
    y = to_categorical(y, num_classes=2)  # Male=0, Female=1
    return X, y

# ============================
# Load Data
# ============================
X, y = load_utkface(r'C:\Users\HP\women-safety-env\UTKFace', target_size=(128, 128))

# ============================
# Build Gender Classifier Model
# ============================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)  # Output: Male (0), Female (1)

gender_model = Model(inputs=base_model.input, outputs=predictions)

# Optionally freeze base model
for layer in base_model.layers:
    layer.trainable = False

# ============================
# Compile Model
# ============================
gender_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================
# Train Model
# ============================
history = gender_model.fit(
    X, y,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# ============================
# Save Trained Model
# ============================
os.makedirs("models", exist_ok=True)
gender_model.save("models/gender_model.h5")
print("✅ Model saved to models/gender_model.h5")
