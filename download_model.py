import tensorflow_hub as hub

# Load a pre-trained ResNet50 model from TF Hub
model_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"
gender_model = hub.KerasLayer(model_url, trainable=False)  # Freeze weights
print("done")