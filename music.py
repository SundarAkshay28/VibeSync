import cv2
import numpy as np
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
from dotenv import load_dotenv
import os
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

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0
    predictions = model.predict(face)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

def main():
    st.write("Starting webcam... Press 'q' to exit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam")
        return

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        emotion = predict_emotion(frame)
        st.write(f"Detected Emotion: {emotion}")

        telugu_tracks = get_recommended_telugu_tracks(emotion)
        st.write(f"Recommended Telugu Songs for {emotion}: {telugu_tracks}")

        # Convert the frame color for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels='RGB', use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    main()
