from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('cnn_image_classifier.h5')
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Image Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Process the uploaded image
        image = Image.open(file).resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict with the model
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
