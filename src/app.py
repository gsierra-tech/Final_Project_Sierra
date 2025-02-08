# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Define the CNN model class (same as used for training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load("notepads\\cnn_baseline.pth", weights_only=True))  # Load the saved model parameters
model.eval()  # Set to evaluation mode

# Streamlit app interface
st.title("Handwritten Digit Classification App")
st.write("Draw a digit (0-9) below, and the app will predict it!")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Black canvas
    stroke_width=10,
    stroke_color="#FFFFFF",  # White drawing
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Preprocess the drawn image
    image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")  # Grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels

    # Clean the image (binarization)
    image_np = np.array(image)
    threshold = 128  # Threshold for binarization
    image_np[image_np < threshold] = 0
    image_np[image_np >= threshold] = 255
    cleaned_image = Image.fromarray(image_np)

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Match training normalization
    ])
    image_tensor = transform(cleaned_image).unsqueeze(0)  # Add batch dimension

    # Display the processed image
    st.image(cleaned_image, caption="Processed Image", width=150)

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)  # Convert to probabilities
        confidence, predicted_class = torch.max(probabilities, 1)

    # Display the prediction with confidence
    st.write(f"**Predicted Digit:** {predicted_class.item()}")
    st.write(f"**Confidence:** {confidence.item():.2f}")

    # Handle low-confidence predictions
    if confidence.item() < 0.7:
        st.warning("The model is not very confident in this prediction. Try drawing more clearly.")
