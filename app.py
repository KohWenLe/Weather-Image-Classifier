import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Load the model and define transforms
model_path = "weather_classification_model.pth"

# Define the model architecture (same as training)
model = models.vgg16(weights='IMAGENET1K_V1')
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 4)  # 4 classes: Cloudy, Rain, Shine, Sunrise
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define class names
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit app layout
st.title("Weather Image Classifier")
st.write("Upload an image, and the model will classify it as either Cloudy, Rain, Shine, or Sunrise.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Process the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Transform and classify image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    # Display prediction
    st.write(f"Prediction: {prediction}")