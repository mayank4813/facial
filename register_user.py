from flask import Flask, request, jsonify
import cv2
import os
import base64
import numpy as np

app = Flask(__name__)

@app.route('/upload-face', methods=['POST'])
def upload_face():
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

@app.route('/train', methods=['POST'])
def train_model():
    data_dir = 'dataset'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
        recognizer.save("trainer.yml")

        # Save labels to file
        with open("labels.txt", "w") as f:
            for id, name in label_map.items():
                f.write(f"{id},{name}\n")

        return jsonify({"status": "success", "message": "Training completed successfully!"})
    else:
        return jsonify({"status": "fail", "message": "No faces found to train."}), 400


if __name__ == "__main__":
    app.run(debug=True)
