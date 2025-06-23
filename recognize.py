import cv2
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Constants
MODEL_PATH = "trained_model/trainer.yml"
LABELS_PATH = "trained_model/labels.txt"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CONFIDENCE_THRESHOLD = 70
MIN_CAPTURE_DURATION = 5  # seconds

def load_labels(label_file):
    """Loads user labels from labels.txt"""
    label_dict = {}
    try:
        with open(label_file, 'r') as f:
            for line in f:
                k, v = line.strip().split(",")
                label_dict[int(k)] = v
    except FileNotFoundError:
        logging.error(f"Label file not found: {label_file}")
    return label_dict

def initialize_recognizer(model_path):
    """Loads trained face recognition model"""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer

def recognize_faces(recognizer, labels):
    """Recognizes multiple faces from live webcam feed"""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        logging.error("Unable to access the camera.")
        return

    logging.info("Camera opened. Scanning for faces... (Minimum 5 seconds)")
    start_time = time.time()

    recognized_names = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            logging.error("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name = labels.get(id_, "Unknown")
                recognized_names.add(name)
                text = f"Authorized: {name}"
                color = (0, 255, 0)
            else:
                recognized_names.add("Unauthorized")
                text = "Unauthorized"
                color = (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        # End after 5 seconds or user presses ESC
        if time.time() - start_time > MIN_CAPTURE_DURATION:
            break
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Print all recognition results
    print("\nScan Result:")
    for name in recognized_names:
        print(f"✅ {name}" if name != "Unauthorized" else "❌ Unauthorized")

def start_face_recognition():
    labels = load_labels(LABELS_PATH)
    recognizer = initialize_recognizer(MODEL_PATH)

    if recognizer is None or not labels:
        logging.error("Recognizer or labels not loaded properly.")
        return

    while True:
        recognize_faces(recognizer, labels)

        again = input("\nDo you want to scan again? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting face recognition.")
            break

if __name__ == "__main__":
    start_face_recognition()
