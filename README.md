Project Title:
Handwritten Digit Classification App

Overview:
This project is a handwritten digit classification app built using a Convolutional Neural Network (CNN). The app allows users to draw digits and predicts them with the trained model.

Features:
    Upload or draw digits (0-9).
    Predict the digit drawn or uploaded.
    Display confidence scores for predictions.
    Real-time predictions with streamlit.

Model Architecture:
The CNN model consists of:
    Two convolutional layers.
    Max pooling layers.
    Fully connected layers.
    A final softmax layer for classification (10 classes: 0-9).

Training Details:
Model Training Process:
    Initial Model: The first version of the model was trained without any data augmentation.
    Improvements:
        Epoch Increase: The model was trained for 20 epochs (instead of 5) to improve performance.
        Data Augmentation:
            Applied random rotation, affine transformations (translation), and normalization.
            Enhanced the training dataset by augmenting data during training.

Performance Improvement:
Iterative Approach:
    Baseline Model: The initial model achieved moderate performance with a basic training process.
    Improved Model:
        Increased epochs from 10 to 20.
        Implemented data augmentation to improve generalization and robustness.

Getting Started:
Prerequisites:

    Python 3.x
    Required libraries:
        torch
        torchvision
        streamlit
        PIL (Pillow)
        numpy

Installation:

    Clone the repository:

git clone https://github.com/your-username/handwritten-digit-classification.git

Navigate to the project directory:

cd handwritten-digit-classification

Install dependencies:

    pip install -r requirements.txt

Running the Streamlit App:

    Ensure your model file cnn_model.pth is available in the project directory.

    Run the Streamlit app:

    streamlit run streamlit_app.py

    Open the app in your browser at: http://localhost:8501

Model Evaluation:

    The model was evaluated by comparing performance on a test dataset (accuracy, precision, recall, F1-score).
    Data augmentation helped improve the robustness and generalization of predictions.

Future Improvements:

    Experimenting with different CNN architectures.
    Further optimizing data augmentation techniques.
    Implementing real-time feedback for drawing.

Acknowledgements:

    Dataset used: MNIST (Modified National Institute of Standards and Technology).

Notes:

    Ensure that cnn_model.pth is placed in the correct directory before running the Streamlit app.
    The model predicts digits from 0-9 with confidence scores for predictions.
