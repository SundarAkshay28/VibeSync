import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
# Load the pre-trained model
model = load_model('facial_expression.h5')

# Define emotion labels
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

def predict_emotion(img):
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0  # Reshape and normalize for the model
    predictions = model.predict(img)
    emotion_index = np.argmax(predictions)
    emotion = emotion_labels[emotion_index]
    return emotion

# Option 1: Process images from a directory
def process_images_from_directory(images_dir):
    for category in os.listdir(images_dir):
        category_path = os.path.join(images_dir, category)
        if not os.path.isdir(category_path):
            continue
        # Loop through each image in the category directory
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            # Read the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Check if the image was loaded correctly
            if img is None:
                print(f"[WARN] Could not load image: {img_path}. Skipping this file.")
                continue

            # Predict emotion
            emotion = predict_emotion(img)
            print(f"Image: {img_name} | Category: {category} | Predicted Emotion: {emotion}")


# Option 2: Real-time emotion detection using webcam
def real_time_emotion_detection():
    cap = cv2.VideoCapture(0)

    # Specify the window mode for displaying the real-time detection
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for emotion prediction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        emotion = predict_emotion(gray)

        # Display the predicted emotion on the frame
        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main Function
if __name__ == "__main__":
    # Choose the mode
    mode = input("Choose mode ('images' for processing images, 'webcam' for real-time detection): ").strip().lower()

    if mode == 'images':
        images_dir = input("Enter the path to the image directory: ").strip()
        if os.path.exists(images_dir):
            process_images_from_directory(images_dir)
        else:
            print("Invalid directory path.")
    elif mode == 'webcam':
        real_time_emotion_detection()
    else:
        print("Invalid mode selected.")
