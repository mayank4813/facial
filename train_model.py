import cv2
import os
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Define paths
DATASET_PATH = 'dataset'
MODEL_PATH = 'trained_model/trainer.yml'
LABELS_PATH = 'trained_model/labels.txt'

# Ensure the trained_model directory exists
os.makedirs("trained_model", exist_ok=True)

def train_model():
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    # Ensure dataset directory exists
    if not os.path.exists(DATASET_PATH):
        logging.error(f"[ERROR] Dataset folder not found at '{DATASET_PATH}'")
        return

    for user_folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, user_folder)
        if os.path.isdir(folder_path):
            label_dict[current_label] = user_folder
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(current_label)
            current_label += 1

    if not faces:
        logging.error("[ERROR] No face images found for training.")
        return

    # Train the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    logging.info(f"[INFO] Model saved to: {MODEL_PATH}")

    # Save label mapping
    with open(LABELS_PATH, 'w') as f:
        for label_id, name in label_dict.items():
            f.write(f"{label_id},{name}\n")
    logging.info(f"[INFO] Labels saved to: {LABELS_PATH}")
    logging.info("[INFO] Model training complete.")

if __name__ == "__main__":
    train_model()
