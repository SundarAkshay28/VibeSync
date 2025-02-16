import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to your dataset folder (data folder)
data_dir = r'E:\Facial_Classification\data'  # Use raw string to avoid escape issues

# Define emotion labels
emotion_labels = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'suprise': 6
}

# Load and preprocess images
def load_data(data_dir, emotion_labels, dataset_type='train'):
    images, labels = [], []
    dataset_folder = os.path.join(data_dir, dataset_type)  # This constructs the path to train or test
    for emotion, label in emotion_labels.items():
        emotion_folder = os.path.join(dataset_folder, emotion)
        print(f"Loading images from: {emotion_folder}")  # Debug line
        if not os.path.exists(emotion_folder):
            print(f"Folder not found: {emotion_folder}")  # Debug line
            continue  # Skip to the next emotion if folder is not found
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(label)
    X = np.array(images).reshape(-1, 48, 48, 1)
    X = X / 255.0
    y = to_categorical(np.array(labels), num_classes=7)
    return X, y


# Load the datasets
X_train, y_train = load_data(data_dir, emotion_labels, dataset_type='E:\\Facial_Classification\\data\\train')
X_test, y_test = load_data(data_dir, emotion_labels, dataset_type='E:\\Facial_Classification\\data\\test')

# Prepare the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=64)

# Save the trained model
model.save('facial_expression_model.h5')

print("Model training complete and saved as 'facial_expression_model.h5'")
