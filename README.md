Project Title:
Optimizing a CNN for Handwritten Digit Recognition: From Model Design to Web Application

Overview:
This project consists of:
    A Convolutional Neural Network (CNN) trained on the MNIST dataset to classify handwritten digits.
    A Streamlit web app where users can draw a digit (0-9) and get real-time predictions using the trained model.

Model Architecture:
The CNN model consists of:
    Two convolutional layers.
    Max pooling layers.
    Fully connected layers.
    A final softmax layer for classification (10 classes: 0-9).

Model Training Details:
    Initial Model: The first version of the model was trained without any data augmentation, with 5 epochs, Optimizer SGD and learning rate 0,01

    Best model: Step by Step changes were tested until identifying the best model (with data augmentation, 10 epochs, Optimizer Adam and Adaptive Learning Rate (LR Scheduler)).

    In the current code, the hyperparamenters are selected for the best model. If wished, parameters can be changed and then the new trained model should be saved under a new name (please see under "Save the model"). For a new model, metrics and confusion matrix can be calculated by indicating the new model name in the appropiate position in the code.


Installation:

Clone the repository:
    git clone https://github.com/gsierra-tech/Final_Project_Sierra.git

Navigate to the project directory:
    cd Final_Project_Sierra

Install dependencies (recommended to activate a virtual environment before):
    pip install -r requirements.txt

Run the code for training the model:
    python CNN.py
    The model will be saved under "cnn_best model.pth"
    For this model a "model_metrics_summary.csv" and a "CNN_best_model_consusion_matrix.csv" will be created


Running the Streamlit App:
    Ensure "cnn_best model.pth" is available in the project directory.

    Run the Streamlit app:
    streamlit run app.py

    The app should automatically open in the browser. If not, go to: http://localhost:8501

    The app allows users to draw a digit (0-9) and predicts it using the trained CNN model.

    If a new CNN model wants to be used (different than the current one: "cnn_best model.pth"), this should be indicated in the app.py

Future Improvements:
    Collect the numbers draw in the app and use them for training the model

Acknowledgements:
    Dataset used: MNIST (Modified National Institute of Standards and Technology).

