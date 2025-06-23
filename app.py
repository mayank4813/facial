from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
import os
import base64
import cv2
import numpy as np
from recognize import initialize_recognizer, load_labels
from datetime import datetime
import shutil
import stat
import time
import errno
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Admin credentials (in a real application, these should be stored securely in a database)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # In production, use a secure password hash

# Constants
MODEL_PATH = "trained_model/trainer.yml"
LABELS_PATH = "trained_model/labels.txt"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CONFIDENCE_THRESHOLD = 70

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def redirect_to_localhost():
    """Redirect IP address access to localhost"""
    if request.host.startswith('127.0.0.1'):
        return redirect(f'http://localhost:{request.host.split(":")[1]}{request.path}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('register'))
        else:
            return render_template('admin_login.html', error='Invalid credentials')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

@app.route('/register')
@admin_required
def register():
    users = get_registered_users()
    return render_template('register.html', active_tab='register', users=users)

@app.route('/recognize')
def recognize():
    return render_template('recognize.html', active_tab='recognize')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('recognize'))
    return render_template('dashboard.html', 
                         username=session['username'],
                         timestamp=session['login_time'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('recognize'))

def get_registered_users():
    """Get list of registered users from dataset directory"""
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        return []
    return [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

def handle_remove_readonly(func, path, exc):
    """Handle read-only files and directories during deletion"""
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        # Change the file to be readable, writable, and executable
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # Try the function again
        func(path)
    else:
        raise

@app.route('/delete-user/<username>')
def delete_user(username):
    try:
        user_dir = os.path.join('dataset', username)
        if os.path.exists(user_dir):
            # First, try to make all files writable
            for root, dirs, files in os.walk(user_dir):
                for dir in dirs:
                    os.chmod(os.path.join(root, dir), stat.S_IRWXU)
                for file in files:
                    os.chmod(os.path.join(root, file), stat.S_IRWXU)
            
            # Try to delete multiple times in case of file system delays
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    shutil.rmtree(user_dir, onerror=handle_remove_readonly)
                    break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(1)  # Wait a second before trying again
            
        # Retrain model if there are remaining users
        remaining_users = get_registered_users()
        if remaining_users:
            train_model()
        else:
            # If no users left, delete the model files
            if os.path.exists(MODEL_PATH):
                try:
                    os.chmod(MODEL_PATH, stat.S_IRWXU)
                    os.remove(MODEL_PATH)
                except Exception as e:
                    print(f"Error deleting model file: {str(e)}")
            
            if os.path.exists(LABELS_PATH):
                try:
                    os.chmod(LABELS_PATH, stat.S_IRWXU)
                    os.remove(LABELS_PATH)
                except Exception as e:
                    print(f"Error deleting labels file: {str(e)}")
        
        return jsonify({
            "status": "success", 
            "message": f"User {username} deleted successfully",
            "remaining_users": remaining_users
        })
    except Exception as e:
        print(f"Error in delete_user: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Could not delete user {username}. Please close any applications that might be using these files and try again."
        }), 500

@app.route('/get-users')
def get_users():
    users = get_registered_users()
    return jsonify({"users": users})

@app.route('/upload-face', methods=['POST'])
def upload_face():
    try:
        data = request.get_json()
        username = data['username']
        count = data['count']
        image_data = data['image']

        # Decode the image
        image_data = image_data.split(',')[1]  # Remove the metadata
        image_data = base64.b64decode(image_data)

        user_dir = f"dataset/{username}"
        os.makedirs(user_dir, exist_ok=True)

        image_path = f"{user_dir}/{count}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_data)

        return jsonify({"status": "success", "message": "Image saved."})
    except Exception as e:
        print(f"Error in upload_face: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data_dir = 'dataset'
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces = []
        labels = []
        label_map = {}
        label_id = 0

        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            label_map[label_id] = person_name
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces_detected = face_cascade.detectMultiScale(img, 1.1, 4)
                for (x, y, w, h) in faces_detected:
                    face = img[y:y+h, x:x+w]
                    faces.append(face)
                    labels.append(label_id)
            label_id += 1

        if faces and labels:
            recognizer.train(faces, np.array(labels))
            
            # Ensure directory exists
            os.makedirs("trained_model", exist_ok=True)
            
            # Save the model and labels
            recognizer.save("trained_model/trainer.yml")
            with open("trained_model/labels.txt", "w") as f:
                for id, name in label_map.items():
                    f.write(f"{id},{name}\n")

            return jsonify({"status": "success", "message": "Training completed successfully!"})
        else:
            return jsonify({"status": "error", "message": "No faces found to train."}), 400

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/start-recognition')
def start_recognition():
    try:
        # Initialize face recognition
        recognizer = initialize_recognizer(MODEL_PATH)
        labels = load_labels(LABELS_PATH)
        
        if recognizer is None or not labels:
            return jsonify({"error": "Face recognition model not properly loaded"}), 500

        # Initialize camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return jsonify({"error": "Unable to access camera"}), 500

        # Capture one frame
        ret, frame = cam.read()
        cam.release()

        if not ret:
            return jsonify({"error": "Failed to capture image"}), 500

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"error": "No faces detected"}), 400

        recognized_people = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name = labels.get(id_, "Unknown")
                recognized_people.append({
                    "name": name,
                    "confidence": round(100 - confidence, 2),
                    "position": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                })
            else:
                recognized_people.append({
                    "name": "Unknown",
                    "confidence": round(100 - confidence, 2),
                    "position": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                })

        # Convert frame to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": "success",
            "recognized_people": recognized_people,
            "frame": frame_base64
        })

    except Exception as e:
        print(f"Recognition error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure the required directories exist
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("trained_model", exist_ok=True)
    # Run on localhost with SSL disabled
    app.run(host='127.0.0.1', port=5000, debug=True)
