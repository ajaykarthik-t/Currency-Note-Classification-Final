# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
import pyttsx3
import threading
import queue

app = Flask(__name__)

# Load the trained model
model = load_model('my_model.keras')

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Create a queue for speech messages
speech_queue = queue.Queue()

# Define class-specific messages
class_messages = {
    0: "You have a 50 rupees note.",
    1: "You have a 500 rupees note.",
    2: "You have a 100 rupees note.",
    3: "You have a 10 rupees note.",
    4: "You have a 20 rupees note.",
    5: "You have a 200 rupees note."
}

def speak_message(message):
    tts_engine.say(message)
    tts_engine.runAndWait()

def speech_worker():
    while True:
        message = speech_queue.get()
        if message is None:
            break
        speak_message(message)
        speech_queue.task_done()

# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = np.array(frame_resized, dtype="float32") / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract ROI
        height, width, _ = frame.shape
        roi_x1 = int(width * 0.3)
        roi_y1 = int(height * 0.3)
        roi_x2 = int(width * 0.7)
        roi_y2 = int(height * 0.7)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Preprocess and predict
        preprocessed_roi = preprocess_frame(roi)
        predictions = model.predict(preprocessed_roi)
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence < 0.6:
            predicted_message = "Please show a currency note."
        else:
            predicted_message = class_messages[predicted_class_index]
            speech_queue.put(predicted_message)

        # Draw ROI and prediction
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.putText(frame, predicted_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)