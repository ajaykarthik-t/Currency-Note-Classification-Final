import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from gtts import gTTS
import os

# Load the trained model
model = load_model('my_model.keras')

# Define class-specific messages
class_messages = {
    0: "You have a 50 rupees note.",
    1: "You have a 500 rupees note.",
    2: "You have a 100 rupees note.",
    3: "You have a 10 rupees note.",
    4: "You have a 20 rupees note.",
    5: "You have a 200 rupees note."
}

# Function to convert text to speech and save as an audio file
def speak_message(message):
    tts = gTTS(text=message, lang='en')
    tts.save("output.mp3")  # Save the audio file
    st.audio("output.mp3", format="audio/mp3")  # Play the audio in Streamlit

# Function to preprocess the image
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = np.array(frame_resized, dtype="float32") / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

# Streamlit UI
st.title("Currency Note Detection for Visually Impaired Users")

# Access webcam
stframe = st.empty()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not access the webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # Extract ROI (central area of the image)
        height, width, _ = frame.shape
        roi_x1 = int(width * 0.3)
        roi_y1 = int(height * 0.3)
        roi_x2 = int(width * 0.7)
        roi_y2 = int(height * 0.7)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Preprocess the ROI
        preprocessed_roi = preprocess_frame(roi)

        # Predict the currency note
        predictions = model.predict(preprocessed_roi)
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence < 0.6:  # Threshold to determine if it's a valid note
            predicted_message = "Please show a currency note."
        else:
            predicted_message = class_messages[predicted_class_index]

        # Draw ROI and prediction on frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.putText(frame, predicted_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display result in Streamlit
        stframe.image(frame, channels="BGR")
        st.success(predicted_message)

        # Announce result using gTTS
        speak_message(predicted_message)

cap.release()
cv2.destroyAllWindows()
