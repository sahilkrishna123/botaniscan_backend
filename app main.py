from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
CORS(app)

# Define class labels
class_labels = {0: 'apple', 1: 'grape', 2: 'tomato'}

# Load the trained model
model_path = '../leaf_classification_model_1_0.h5'
model = load_model(model_path)
print(f"Model loaded successfully from {model_path}")

def preprocess_image_from_pil(img):
    img = img.resize((500, 500))  # Resize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values
    return img_array

def predict_image(img, model):
    preprocessed_image = preprocess_image_from_pil(img)
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    predicted_class = class_labels[class_index]
    confidence = prediction[0][class_index]
    return predicted_class, confidence

@app.route('/')
def hello():
    return 'Hello, Flask!'

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if the 'image' key is present in request.files
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the file from request.files
        file = request.files['image']

        # Open the image using PIL
        img = Image.open(io.BytesIO(file.read()))

        # Perform prediction
        predicted_class, confidence = predict_image(img, model)

        # Return the result
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence) * 100
        }
        return jsonify(result)

    except ValueError as ve:
        print("ValueError:", str(ve))
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify({"error": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
