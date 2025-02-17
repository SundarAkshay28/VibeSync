import cv2
import numpy as np
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
st.title("Emotion Detection with Telugu Song Recommendations")

# Spotify API credentials
clientid = os.getenv('CLIENT_ID')
clientsecret = os.getenv('CLIENT_SECRET')
REDIRECT_URI = 'http://localhost:8000/callback'

# Authenticate with Spotify using OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=clientid,
    client_secret=clientsecret,
    redirect_uri=REDIRECT_URI,
    scope="user-library-read user-read-playback-state user-modify-playback-state"
))

# Load pre-trained model
model = load_model('facial_expression.h5')

emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

def get_recommended_telugu_tracks(emotion):
    mood_to_genre_telugu = {
        'happy': 'Telugu upbeat',
        'sad': 'Telugu sad',
        'fear': 'Telugu emotional',
        'surprise': 'Telugu party',
        'neutral': 'Telugu melody',
        'angry': 'Telugu rock',
        'disgust': 'Telugu folk'
    }
    genre = mood_to_genre_telugu.get(emotion, 'Telugu')
    results = sp.search(q=f'{genre}', type='track', limit=5, market='IN')
    return [track['name'] + ' - ' + track['artists'][0]['name'] for track in results['tracks']['items']]

def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0
    predictions = model.predict(face)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

def main():
    st.write("Upload an image to detect emotion and get Telugu song recommendations.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict emotion
        emotion = predict_emotion(image)
        st.write(f"**Detected Emotion:** {emotion}")

        # Get recommended Telugu songs
        telugu_tracks = get_recommended_telugu_tracks(emotion)
        st.write(f"**Recommended Telugu Songs for {emotion}:**")
        for track in telugu_tracks:
            st.write(f"- {track}")

if __name__ == '__main__':
    main()
