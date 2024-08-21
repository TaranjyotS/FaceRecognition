import numpy as np

from keras.api.models import load_model
from keras.api.preprocessing import image


# Load the pre-trained Keras model
model = load_model('./keras_model/emotion_detection_model.keras')  # Update with your model path

# Define emotion labels corresponding to the output classes
emotion_labels: list[str] = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


def preprocess_image(img_path: str, target_size=(48, 48)):
    """
    Load and preprocess the image to fit the model's input shape.

    Args:
        img_path: Path to the image for which to detect the emotion.
        target_size: Target size of the image.
    
    Returns:
        img_array: A batch dimension to the image array.
    """
    # Load image
    img = image.load_img(img_path, color_mode='grayscale', target_size=target_size)
    
    # Convert image to array
    img_array = image.img_to_array(img)
    
    # Normalize image
    img_array = img_array / 255.0
    
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_emotion(img_path: str) -> str:
    """
    Predict the emotion of the face in the image.

    Args:
        img_path: Path to the image for which to detect the emotion.

    Returns:
        emotion: The emotion detected from the image.
    """
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the index of the highest probability
    emotion_index = np.argmax(prediction)
    
    # Map index to emotion label
    emotion = emotion_labels[emotion_index]
    
    return emotion

image_path = input('Provide the path to image: ')
predicted_emotion = predict_emotion(image_path)
print(f"Predicted Emotion: {predicted_emotion}")
