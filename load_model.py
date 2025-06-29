from tensorflow.keras.applications import ResNet50

# Load a pretrained ResNet50 model
gender_model = ResNet50(weights="imagenet")  

# Save it as gender_model.h5 for future use
gender_model.save("gender_model.h5")  