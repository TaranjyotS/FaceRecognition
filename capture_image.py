import cv2
import os


# Name of the directory where images will be saved.
name = input("Enter your name: ")

# Directory to save captured images
output_dir = f'./images/{name}'
os.makedirs(output_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Number of images to capture
num_images = 5
count = 0

print(f"Capturing {num_images} images. Press 's' to save an image and 'q' to quit at any time.")

while count < num_images:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow('Image Capture', frame)

    # Wait for user input
    key = cv2.waitKey(1)
    if key == ord('s'):  # Save image when 's' is pressed
        image_name = os.path.join(output_dir, f'image_{count}.jpg')
        cv2.imwrite(image_name, frame)
        count += 1
        print(f"Image {count} saved as {image_name}")
    elif key == ord('q'):  # Quit when 'q' is pressed
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Captured {count} images.")
