import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the path to your dataset folder
data_dir = r'E:\Facial_Classification\data\test'

# Define emotion labels
emotion_labels = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6  # Corrected spelling
}

# Function to load and preprocess images
def load_data(data_dir, emotion_labels):
    images, labels = [], []
    for emotion, label in emotion_labels.items():
        # Construct the path for each emotion folder
        emotion_folder = os.path.join(data_dir, emotion)
        print(f"\n[INFO] Checking directory: {emotion_folder}")

        # Check if the directory exists
        if not os.path.exists(emotion_folder):
            print(f"[ERROR] Directory not found: {emotion_folder}")
            continue

        # List the contents of the folder
        files = os.listdir(emotion_folder)
        print(f"[INFO] Found {len(files)} files in '{emotion}' folder")

        # Load images from the folder
        for img_name in files:
            img_path = os.path.join(emotion_folder, img_name)

            # Ensure the file is an image
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"[WARNING] Skipping non-image file: {img_name}")
                continue

            # Read and preprocess the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
            else:
                print(f"[WARNING] Failed to load image: {img_path}")

    # Ensure we have loaded some data
    if not images:
        raise ValueError("[ERROR] No images were found in the dataset directories.")

    # Convert lists to numpy arrays
    X = np.array(images).reshape(-1, 48, 48, 1)
    X = X / 255.0  # Normalize pixel values
    y = to_categorical(np.array(labels), num_classes=len(emotion_labels))

    return X, y


# Load and prepare the data
try:
    X, y = load_data(data_dir, emotion_labels)
except ValueError as e:
    print(e)
    exit(1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=64)

# Save the trained model
model.save('facial_expression.h5')
print("Model saved as 'facial_expression.h5'")
