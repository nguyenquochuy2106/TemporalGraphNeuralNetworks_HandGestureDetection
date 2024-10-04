# streamlit_app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_training import TGNNModel  # Ensure this is your model definition
import os

# Load your trained model
def load_model(model_path):
    model = TGNNModel(num_features=..., num_classes=...)  # Update with your actual numbers
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define a function to process images for prediction
def process_image(image):
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Adjust as necessary
        transforms.ToTensor(),
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def main():
    st.title("Gesture Recognition with TGNN")
    
    model_path = 'saved_models/best_model.pth'
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image for prediction
        processed_image = process_image(image)

        # Make prediction (update as necessary based on your model's input)
        with torch.no_grad():
            output = model(processed_image)
            predicted_class = torch.argmax(output, dim=1).item()  # Get predicted class index

        # Display the prediction
        st.write(f"Predicted class: {predicted_class}")  # Map class index to label if necessary

if __name__ == "__main__":
    main()
