import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reinitialize the model architecture with the correct output features
model = models.resnet18(weights=None)  # Use pretrained=False to avoid loading ImageNet weights
model.fc = nn.Linear(in_features=512, out_features=4)  # Adjust to match the number of output classes in your saved model
model = model.to(device)

# Load the saved state dictionary
model_path = 'model.pth'  # Replace with the correct path to your model
state_dict = torch.load(model_path, map_location=device)  # Load the state dictionary
model.load_state_dict(state_dict)  # Load it into the model
model.eval()  # Set the model to evaluation mode

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Prediction function
def predict_image(image):
    """
    Predicts the class of an image.

    Args:
        image (PIL.Image): Image to be classified.

    Returns:
        str: Predicted class label.
    """
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)

        # Map predicted class to label
        class_labels = {0: 'Blight', 1: 'Common_Rust', 2: 'Gray_Leaf_Spot', 3: 'Healthy'}  # Replace with actual class labels
        return class_labels[predicted_class.item()]

    except Exception as e:
        return f"Error processing the image: {e}"

# Streamlit app
st.title("Corn Leaf Disease Detection")
st.write("Upload an image of a corn leaf to detect its health status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")
    # Predict the class
    result = predict_image(image)
    st.write(f"Predicted Class: **{result}**")
