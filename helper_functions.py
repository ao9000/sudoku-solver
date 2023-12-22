"""
    Contains helper functions for the training and evaluating of the model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import tensorflow as tf
import cv2


def get_mnist_dataset():
    """
    Gets the mnist dataset and modifies the dataset by removing digit 0 images and labels since we will not be needing it

    :return: type: tuple of lists
    Returns the tuple of lists for images and labels for training the validation sets
    """
    # Load MNIST dataset
    # 60,000 28x28 grayscale Train images of the 10 digits, 10,000 Test images
    # x_train -> images in numpy array, y_train -> corresponding labels
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Remove 0 from train & test datasets, since we do not need them
    # Get all 0 indexes in train
    train_remove_indexes = [index for index, label in enumerate(y_train) if label == 0]

    # Remove items in train where match index
    x_train = np.asarray([image for index, image in enumerate(x_train) if index not in train_remove_indexes])
    y_train = np.asarray([label for index, label in enumerate(y_train) if index not in train_remove_indexes])

    # Remove 0 from train & test datasets, since we do not need them
    # Get all 0 indexes in test
    test_remove_indexes = [index for index, label in enumerate(y_test) if label == 0]

    # Remove items in test where match index
    x_test = np.asarray([image for index, image in enumerate(x_test) if index not in test_remove_indexes])
    y_test = np.asarray([label for index, label in enumerate(y_test) if index not in test_remove_indexes])

    return (x_train, y_train), (x_test, y_test)


def preprocess_train_dataset(data):
    """
    Prepares the training dataset for model training

    Process:
    1. Reshapes the images the required size for model ingestion
    2. Normalize the pixel values to allow the model to train faster

    :param data: type: numpy.ndarray
    Training images

    :return: type: numpy.ndarray
    Preprocessed training images
    """
    # Reshape to be samples*pixels*width*height & convert images into single channel
    data = data.reshape((data.shape[0], 28, 28, 1))

    # Prepare image for modelling
    # Normalize pixel values 0-1
    data_norm = data.astype('float32')
    data_norm = data_norm / 255.0

    return data_norm


def preprocess_train_label(label):
    """
    Preprocess the training labels for model ingestion

    :param label: type: numpy.ndarray
    Training labels of the respective images

    :return: type: numpy.ndarray
    Preprocessed training labels
    """
    # One hot Code
    # Converts a class vector (integers) to binary class matrix
    label = to_categorical(label - 1, 9)

    return label


def sudoku_cells_reduce_noise(digit_inv):
    """
    Noises in the cell images can impact the accuracy of the model
    Perform some image preprocessing on the extracted cell digits images to improve accuracy

    Process:
    1. Detect contour of the digit in the cell
    2. Filter the contours over5 pixels area
    3. Get the largest contour (Assuming the largest contour is the digit)
    4. Crop the digit
    5. Resize the digit to a standard size
    6. Create a black mat and paste the cropped digit onto it

    :param digit_inv: type: numpy.ndarray
    Binary image of the detected cell

    :return: type: numpy.ndarray
    Binary image of the digit after noise reduction
    """
    # Eliminate surrounding noise
    # Detect contours
    cnts, hierarchy = cv2.findContours(digit_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours over 5 pixel square area
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 5]

    # Check if any contour is detected
    if cnts:
        # Sort to largest contour (Digit)
        cnt = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        # Get coordinates, width, height of contour
        x, y, width, height = cv2.boundingRect(cnt)

        # Create buffer for crop
        crop_buffer = 1
        # Decrement crop buffer if buffer goes out of bounds of image
        while (y-crop_buffer) < 0 or (x-crop_buffer) < 0 or (y+height+crop_buffer) > digit_inv.shape[0] or (x+width+crop_buffer) > digit_inv.shape[1]:
            crop_buffer -= 1

            if crop_buffer == 0:
                break

        # Crop area
        digit_inv = digit_inv[y-crop_buffer:y + height+crop_buffer, x-crop_buffer:x + width+crop_buffer]
        # Update height & width
        height = height + (crop_buffer*2)
        width = width + (crop_buffer*2)

        # Create a black mat
        new_digit_inv = np.zeros((28, 28), np.uint8)

        # Standardize all image sizes
        # Maintain aspect ratio, resize via height or width (Whichever is bigger)
        resized_target_height_width = 17

        if height > width:
            # Height is larger
            aspect_ratio = resized_target_height_width / float(height)
            new_dimensions = (int(width * aspect_ratio), resized_target_height_width)
        else:
            # Width is larger
            aspect_ratio = resized_target_height_width / float(width)
            new_dimensions = (resized_target_height_width, int(height * aspect_ratio))

        # Don't allow any dimension to be 0, will result in error
        if new_dimensions[0] <= 3 or new_dimensions[1] <= 3:
            new_dimensions = (resized_target_height_width, resized_target_height_width)

        # Check if original image is larger is smaller
        if height > resized_target_height_width:
            # Shrink
            digit_inv = cv2.resize(digit_inv, new_dimensions, interpolation=cv2.INTER_AREA)
        else:
            # Expand
            digit_inv = cv2.resize(digit_inv, new_dimensions, interpolation=cv2.INTER_CUBIC)

        # Update width & height
        height, width = digit_inv.shape

        # Paste detected contour in the middle to center image
        new_digit_inv[14-height//2:14-height//2+height, 14-width//2:14-width//2+width] = digit_inv

        return new_digit_inv
    else:
        # No contour detected
        return None


def build_model():
    """
    Builds a model for digit classification
    Reference: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

    :return: type: tensorflow.python.keras.engine.sequential.Sequential
    The model used for digit classification
    """
    # Build the CNN Model for digit classification purposes
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def plot_accuracy_graph(history):
    """
    Plot the model accuracy graph across epochs during training
    And save as image

    :param history: type: tensorflow.python.keras.callbacks.History
    History object for the training progress records
    """
    # Set x axis to 1 step
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot training graphs
    # Accuracy graph
    # Plot
    plt.plot(history.history['accuracy'], color='blue', label="Train accuracy")
    plt.plot(history.history['val_accuracy'], color='orange', label="Val accuracy")
    # Set labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    # Save plot
    plt.savefig("models/accuracy.png")


def plot_loss_graph(history):
    """
    Plot the model loss graph across epochs during training
    And save as image

    :param history: type: tensorflow.python.keras.callbacks.History
    History object for the training progress records
    """
    # Set x axis to 1 step
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Loss graph
    # Plot
    plt.plot(history.history['loss'], color='blue', label="Train loss")
    plt.plot(history.history['val_loss'], color='orange', label="Val loss")
    # Set labels
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    # Save plot
    plt.savefig("models/loss.png")
