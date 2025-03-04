import cv2
import face_recognition
import os
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai

# Initialize face database
FACE_DATABASE_PATH = "face_database"
os.makedirs(FACE_DATABASE_PATH, exist_ok=True)

known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load known faces from directory."""
    known_face_encodings.clear()
    known_face_names.clear()
    for filename in os.listdir(FACE_DATABASE_PATH):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(FACE_DATABASE_PATH, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()

# Initialize webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)
engine.setProperty('volume', 1.0)

# Configure Google Gemini API
genai.configure(api_key="AIzaSyC9jP5kuFHY9Z1puuEqYDusjGVgqJEWIKc")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
convo = model.start_chat(history=[])

def get_response(user_input):
    try:
        convo.send_message(user_input)
        return convo.last.text
    except Exception as e:
        return "Sorry, I couldn't process that."

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source)
    recognizer.dynamic_energy_threshold = 3000
    
    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                print("Unknown face detected! Press 's' to save and register.")
                cv2.putText(frame, "Press 's' to register", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    new_name = input("Enter name: ")
                    image_path = os.path.join(FACE_DATABASE_PATH, f"{new_name}.jpg")
                    cv2.imwrite(image_path, frame)
                    load_known_faces()
                    name = new_name
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow("Video", frame)
        
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5.0)
            response = recognizer.recognize_google(audio)
            print(f"User said: {response}")
            
            if response.lower() in ["exit", "stop", "quit", "bye", "goodbye"]:
                print("Exiting the program. Goodbye!")
                engine.say("Goodbye!")
                engine.runAndWait()
                break
            
            response_from_gemini = get_response(response)
            print("Gemini AI:", response_from_gemini)
            engine.say(response_from_gemini)
            engine.runAndWait()
        except sr.UnknownValueError:
            print("Didn't recognize anything.")
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")
        
        if cv2.waitKey(1) & 0xFF == ord("x"):
            break

video_capture.release()
cv2.destroyAllWindows()