import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from keras.api.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.api.models import Sequential
from keras.api.utils import to_categorical
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (48, 48)  # Size of the images
BATCH_SIZE = 32


warnings.filterwarnings('ignore')


def load_data():
    """Loads the cohn-kanade dataset and pre-processes it to do the following:
        - Resize images to a consistent size.
        - Convert images to grayscale (if the dataset is not already in grayscale).
        - Normalize pixel values to the range [0, 1].

    Returns:
        X: Reshaped image.
        y: A binary matrix representation of the input as a NumPy array.
        num_classes: Number of classes in our dataset.
    """
    data = pd.read_csv('./dataset/cohn-kanade-dataset.csv')
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32')).values
    emotions = data['emotion'].values
    
    # Determine the number of unique classes
    num_classes = len(np.unique(emotions))
    print(f"Number of unique classes: {num_classes}")

    # Assuming grayscale images; reshape to (height, width, channels)
    X = np.array([pixel.reshape(48, 48, 1) for pixel in pixels])  # For grayscale images
    # X = np.array([pixel.reshape(48, 48, 3) for pixel in pixels])  # For RGB images

    y = to_categorical(emotions, num_classes=num_classes)
    
    return X, y, num_classes


def prepare_data_generators(X_train, y_train, X_test, y_test):
    """Prepares the data for training and testing."""
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator()

    # Prepare data generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

    return train_generator, test_generator


def create_model(num_classes: int):
    """Builds a simple CNN model.
    
    Args:
        num_classes: Number of classes in our dataset.

    Returns:
        A CNN model.
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # For grayscale images
        # Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 3)),  # For RGB images
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, train_generator, test_generator):
    """Trains the model and plots the training history."""
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


def do_it():
    """Script entry point."""
    # Load data
    X, y, num_classes = load_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_generator, test_generator = prepare_data_generators(X_train, y_train, X_test, y_test)
    
    # Create and train the model
    model = create_model(num_classes)
    train_model(model, train_generator, test_generator)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')

    # Save the model
    model.save('./keras_model/emotion_detection_model.keras')


if __name__ == '__main__':
    do_it()
