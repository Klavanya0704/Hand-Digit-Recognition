# Handwritten Digit Recognition Web App

This project is a web application that recognizes handwritten digits (0-9). It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and a web interface built with Streamlit.

## Features
- **Draw a Digit**: Use the on-screen canvas to draw a digit.
- **Upload Image**: Upload an image of a digit.
- **Real-time Prediction**: Displays the predicted digit and confidence score.
- **Preprocessing**: Automatically handles resizing, centering, and normalization to match MNIST format.

## Setup & Running Locally

### Prerequisites
- Python 3.7+
- pip

### Installation
1.  Clone this repository or download the files.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Training the Model
(Optional) If you want to retrain the model:
```bash
python train_model.py
```
This will generate a `model.h5` file.

### Running the App
Start the Streamlit application:
```bash
streamlit run app.py
```
The app will open in your default web browser (usually at `http://localhost:8501`).

## Running on Google Colab

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Create a new notebook.
3.  Upload `train_model.py`, `app.py`, and `requirements.txt` to the Colab files section.
4.  Run the following commands in a code cell to install dependencies and train the model:
    ```python
    !pip install -r requirements.txt
    !python train_model.py
    ```
5.  To run the Streamlit app in Colab, you need a tunnel (since Colab runs on a remote VM). We can use `localtunnel`:
    ```python
    !pip install streamlit
    !npm install localtunnel
    
    # Run streamlit in the background
    !streamlit run app.py &>/content/logs.txt &
    
    # Expose the port
    !npx localtunnel --port 8501
    ```
    *Click the url provided in the output to access the app.*

## Project Structure
- `train_model.py`: Script to train the CNN model on MNIST dataset.
- `app.py`: The Streamlit web application.
- `requirements.txt`: List of Python dependencies.
- `model.h5`: The trained model file (generated after training).
