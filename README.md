# Hand_digit_recognition
## Overview
This project is a **Handwritten Digit Recognizer** that allows users to draw digits in real time using a **Tkinter GUI**, and the system predicts the digit using a **Convolutional Neural Network (CNN)** model. The CNN is trained on the MNIST dataset and utilizes **TensorFlow** and **Keras** for building and training the model. The model is then used to classify the handwritten digits drawn by users on the GUI canvas.

## Features
- **Real-time Drawing and Recognition**: Draw digits in the GUI and get real-time recognition.
- **Deep Learning Model**: Uses a pre-trained CNN model to recognize digits.
- **Easy to Use**: User-friendly interface with the ability to clear the canvas and retry.

## Files in the Project

1. **`gui_digit_recognizer.py`**  
   This is the main script that runs the Tkinter GUI for real-time digit recognition. It creates a canvas for users to draw digits and uses the pre-trained CNN model to recognize the drawn digits.

2. **`train_digit_recognizer.py`**  
   This script is responsible for training the CNN model using the **MNIST dataset**. It defines the CNN architecture, compiles the model, trains it on the dataset, and saves the model to `mnist.h5` for use in the GUI.

3. **`mnist.h5`**  
   This file contains the trained CNN model saved in **HDF5 format**. It is loaded by the `gui_digit_recognizer.py` script to make predictions on the drawn digits.

## Requirements
To run this project, you need to install the following libraries:
- `tensorflow`
- `keras`
- `numpy`
- `Pillow` (for handling image formats)
- `Tkinter` (comes pre-installed with Python)

You can install these dependencies using:
```bash
pip install tensorflow keras numpy Pillow
```
How It Works
Model Training:
The CNN model is trained on the MNIST dataset using the train_digit_recognizer.py script. The model is built using TensorFlow and Keras, with several convolutional layers, pooling layers, and fully connected layers. The model is then saved as mnist.h5.

Real-time Prediction:
The gui_digit_recognizer.py script sets up a Tkinter GUI with a canvas where users can draw digits. The drawn image is resized to 28x28 pixels (to match the input shape of the CNN), normalized, and passed to the model for prediction. The predicted digit is then displayed.

1.Running the Project
Train the Model (optional): If you want to train the model from scratch, run the train_digit_recognizer.py script:
```bash
python train_digit_recognizer.py
```

2.Run the GUI: To start the Tkinter GUI and recognize digits in real-time, run the gui_digit_recognizer.py script:
```bash
python gui_digit_recognizer.py
```

3.Draw and Predict:

A window will open with a blank canvas.
Draw a digit using your mouse.
Once you are done, the model will predict the digit and display the result.
Model Architecture
The CNN model used for digit recognition has the following layers:

## Model Architecture
- Input Layer: 28x28 grayscale images.
- Convolutional Layers: To extract spatial features from the images.
- Max Pooling Layers: For down-sampling the feature maps.
- Fully Connected Layers: For classification.
- Output Layer: Softmax output for predicting one of the 10 digit classes (0-9).
## Example Workflow
- Open the GUI application by running gui_digit_recognizer.py.
- Draw any digit (0-9) on the canvas.
- The CNN model processes the drawing and displays the predicted digit.
## Future Improvements
Add functionality to save and load new drawings for further analysis.
Improve accuracy for more complex drawings and real-world digit styles.
Implement additional features like multi-digit recognition or arithmetic operations.
