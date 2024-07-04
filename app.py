from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
loaded_inception = tf.keras.models.load_model('inception.keras')
labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def preprocess(image):
    image = image.convert('RGB')
    image = image.resize((180, 180))  # Resize to the expected size
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_image(image):
    processed_image = preprocess(image)
    prediction = loaded_inception.predict(processed_image)
    label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return f'Prediction: {label} ({confidence:.2f})'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = Image.open(file)
            prediction = predict_image(image)
            return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
