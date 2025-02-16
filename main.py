import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths and parameters
data_path = './data/FER2013/'
img_size = 48
num_classes = 7  # 7 expressions

# Initialize lists to store images and labels
images, labels = [], []

# Load and preprocess data
for emotion_folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, emotion_folder)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(int(emotion_folder))  # Use folder name as the label

# Convert lists to numpy arrays and normalize images
images = np.array(images) / 255.0
images = images.reshape(-1, img_size, img_size, 1)
labels = to_categorical(np.array(labels), num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
