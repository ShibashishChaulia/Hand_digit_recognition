# file1.py - Updated to encapsulate the code inside a function

import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

def perform_prediction(img_path):
    # Your code from file1.py
    # Loading the dataset
    mnist = tf.keras.datasets.mnist

    # Loading the trained model
    model = load_model('mnist.h5')

    # ... (Rest of the code for data preprocessing, model creation, etc.)
    # Divide into training and test dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data(path="mnist.npz")

x_train.shape
# (60000, 28, 28)

plt.imshow(x_train[0])
plt.show()

plt.imshow(x_train[0], cmap = plt.cm.binary)

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
plt.imshow(x_train[0], cmap = plt.cm.binary)

# verify that there is a proper label for the image
print(y_train[0])
# 5

IMG_SIZE=28
# -1 is a shorthand, which returns the length of the dataset
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Training Samples dimension", x_trainr.shape)
print("Testing Samples dimension", x_testr.shape)
# Training Samples dimension (60000, 28, 28, 1)
# Testing Samples dimension (10000, 28, 28, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Creating the network
model = Sequential()

### First Convolution Layer
# 64 -> number of filters, (3,3) -> size of each kernal,
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) # For first layer we have to mention the size of input
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Second Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Third Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Fully connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

### Fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

### Fully connected layer 3, output layer must be equal to number of classes
model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x_trainr, y_train, epochs=5, validation_split = 0.3)

# Evaluating the accuracy on the test data
test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Test Loss on 10,000 test samples", test_loss)
print("Test Accuracy on 10,000 test samples", test_acc)

predictions = model.predict([x_testr])
print(predictions)

plt.imshow(x_test[8])
print(np.argmax(predictions[8]))

    # Assuming the code for prediction and image processing is in this function
def predict_new_image(img_path):
        # Read the image using cv2
        img = cv2.imread(img_path)

        # Your image processing and prediction code here

        return predicted_label  # Assuming predicted_label contains the predicted value

return predict_new_image(img_path)
