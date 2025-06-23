from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import os
import logging
import base64
from PIL import Image
from io import BytesIO

routes = Blueprint('routes', __name__)

# Logging setup
logging.basicConfig(level=logging.INFO)

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@routes.route('/start-registration', methods=['POST'])
def start_registration():
    data = request.get_json()
    username = data.get('username')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    return render_template('register.html', username=username)

@routes.route('/upload-face', methods=['POST'])
def upload_face():
    username = request.args.get('username')
    count = request.args.get('count')

    if not username or not count:
        return jsonify({"error": "Missing parameters"}), 400

    try:
        count = int(count)
    except ValueError:
        return jsonify({"error": "Count must be an integer"}), 400

    data = request.get_json()
    image_data = data.get('image')
    if not image_data:
        return jsonify({"error": "No image data received"}), 400

    try:
        image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes)).convert('L')

        user_dir = os.path.join('dataset', username)
        os.makedirs(user_dir, exist_ok=True)

        from werkzeug.utils import secure_filename
        filename = secure_filename(f'{username}_{count}.jpg')
        file_path = os.path.join(user_dir, filename)

        img.save(file_path)

        logging.info(f"[INFO] Saved face image: {file_path}")
        return jsonify({"status": "success", "path": file_path})
    except Exception as e:
        logging.error(f"[ERROR] Failed to save image: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
