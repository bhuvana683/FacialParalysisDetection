from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from PIL import Image
from io import BytesIO
from preprocess import preprocess_image

app = Flask(__name__)
model = load_model('paralysis_detection_cnn_lstm_model.keras')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    confidence_scores = prediction[0]
    labels = ['Normal', 'Mild', 'Moderate', 'Severe']
    prediction_dict = {labels[i]: float(confidence_scores[i]) for i in range(len(labels))}
    predicted_class = np.argmax(confidence_scores)
    confidence = float(np.max(confidence_scores))
    return prediction_dict, predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction_dict, label, confidence = predict_image(filepath)
        labels = ['Normal', 'Mild', 'Moderate', 'Severe']
        return jsonify({
            'prediction': labels[label], 
            'confidence': confidence,
            'class_confidence': prediction_dict
        })
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        header, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)

        img = Image.open(BytesIO(decoded)).convert('RGB')
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam.jpg')
        img.save(img_path)

        prediction_dict, label, confidence = predict_image(img_path)
        labels = ['Normal', 'Mild', 'Moderate', 'Severe']
        return jsonify({
            'prediction': labels[label], 
            'confidence': confidence,
            'class_confidence': prediction_dict
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process webcam image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)