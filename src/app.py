# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import center_of_mass

# Define the CNN model (same as used in training)
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

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load("notepads/cnn_baseline.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("Handwritten Digit Recognition App ✍️")
st.write("Draw a digit (0-9) below, and the model will predict it!")

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=10,
    stroke_color="#FFFFFF",  # White pen
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

def center_image(img_np):
    """Centers the digit in the image by shifting it to the middle."""
    if np.sum(img_np) == 0:  # Check if the image is empty
        return img_np  # Return as is to avoid NaN error

    row_sums = img_np.sum(axis=1)
    col_sums = img_np.sum(axis=0)
    row_com, col_com = center_of_mass(img_np)

    if np.isnan(row_com) or np.isnan(col_com):  # Extra check for NaN values
        return img_np  # Return the original image without shifting

    row_shift = int(img_np.shape[0] // 2 - row_com)
    col_shift = int(img_np.shape[1] // 2 - col_com)

    return np.roll(img_np, shift=(row_shift, col_shift), axis=(0, 1))

def preprocess_image(image):
    """Preprocess the drawn image: grayscale, resizing, centering, and normalization"""
    image = Image.fromarray(image.astype("uint8")).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels

    # Convert to NumPy array and normalize
    image_np = np.array(image).astype(np.float32)

    # Centering the digit
    image_np = center_image(image_np)

    # Normalize pixel values (like MNIST)
    image_np = image_np / 255.0  # Scale to [0,1]
    image_np = (image_np - 0.1307) / 0.3081  # Normalize to MNIST mean/std

    # Convert to PyTorch tensor and reshape
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image_tensor

if canvas_result.image_data is not None:
    # Process the image for model prediction
    processed_image = preprocess_image(canvas_result.image_data)

    # Convert the processed image to [0,1] range for Streamlit display
    processed_np = processed_image.squeeze().numpy()
    processed_np = (processed_np - processed_np.min()) / (processed_np.max() - processed_np.min() + 1e-8)  # Avoid divide by zero

    st.image(processed_np, caption="Processed Image", width=150)

    # Make a prediction
    with torch.no_grad():
        output = model(processed_image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Display the result
    st.write(f"### **Predicted Digit:** {predicted_class.item()}")
    st.write(f"**Confidence:** {confidence.item():.2f}")

    if confidence.item() < 0.7:
        st.warning("🤔 The model is unsure. Try drawing more clearly.")
