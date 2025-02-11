# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
import pyttsx3
import threading
import queue
import time

app = Flask(__name__)

# Load the trained model globally
print("Loading model...")
model = load_model('my_model.keras')
print("Model loaded successfully!")

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Create a queue for speech messages
speech_queue = queue.Queue()

# Store the last prediction time
last_prediction_time = 0
PREDICTION_INTERVAL = 0.5  # Make prediction every 0.5 seconds

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
    try:
        tts_engine.say(message)
        tts_engine.runAndWait()
    except RuntimeError:
        print("TTS Engine error - reinitializing")
        tts_engine.init()

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
    # Ensure the frame is in BGR format
    if len(frame.shape) == 2:  # If grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = np.array(frame_resized, dtype="float32") / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.last_prediction = None
        self.last_prediction_time = 0
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        global last_prediction_time
        success, frame = self.video.read()
        if not success:
            return None

        # Extract ROI (central area of the image)
        height, width, _ = frame.shape
        roi_x1 = int(width * 0.3)
        roi_y1 = int(height * 0.3)
        roi_x2 = int(width * 0.7)
        roi_y2 = int(height * 0.7)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        current_time = time.time()
        if current_time - last_prediction_time >= PREDICTION_INTERVAL:
            # Preprocess the ROI
            preprocessed_roi = preprocess_frame(roi)

            # Predict the currency note
            try:
                predictions = model.predict(preprocessed_roi, verbose=0)
                predicted_class_index = np.argmax(predictions)
                confidence = np.max(predictions)

                if confidence < 0.6:  # Threshold to determine if it's a valid note
                    predicted_message = "Please show a currency note."
                else:
                    predicted_message = class_messages[predicted_class_index]
                    # Only add to speech queue if prediction changed
                    if predicted_message != self.last_prediction:
                        speech_queue.put(predicted_message)
                        self.last_prediction = predicted_message

                # Draw ROI and prediction on frame
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_message, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Prediction error: {str(e)}")
                predicted_message = "Error in prediction"

            last_prediction_time = current_time

        # Encode frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

video_camera = None

def generate_frames():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    
    while True:
        frame = video_camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, threaded=True)