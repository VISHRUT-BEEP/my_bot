import requests
import cv2
import numpy as np
import speech_recognition as sr
import time
from picamera2 import Picamera2

SERVER_URL = "http://YOUR_LAPTOP_IP:5000/process"

def capture_image():
    """Capture image using Pi Camera and send to server."""
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)  # Allow camera to adjust
    image = picam2.capture_array()
    picam2.stop()

    _, buffer = cv2.imencode(".jpg", image)
    image_data = buffer.tobytes().hex()

    response = requests.post(SERVER_URL, json={"face_image": image_data})
    print("Server Response:", response.json()["response"])

def capture_audio():
    """Record audio and send to server."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print("You said:", text)

            response = requests.post(SERVER_URL, json={"voice_text": text})
            print("Server Response:", response.json()["response"])
        except:
            print("Could not recognize speech.")

# Run continuously
while True:
    capture_image()
    capture_audio()
    time.sleep(5)
