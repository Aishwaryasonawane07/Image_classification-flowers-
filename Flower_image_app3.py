from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
import os

app = Flask(__name__)

loaded_model_2 = load_model("inception_2.h5")

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


def preprocess(img_path):
    img = keras_image.load_img(img_path, target_size=(180, 180))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    processed_image = preprocess(img_path)
    prediction = loaded_model_2.predict(processed_image)
    label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "Prediction": label,
        "Accuracy": f"{confidence * 100:.2f}%"
    }

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file selected'}), 400
    

    elif not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'File must be in JPG, PNG, or JPEG format'}), 400
    
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    img_path = os.path.join(temp_dir, file.filename)
    file.save(img_path)

    result = predict_image(img_path)
    os.remove(img_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
