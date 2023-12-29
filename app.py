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
class_labels_of_classification = {0: 'apple', 1: 'grape', 2: 'tomato'}
class_labels_of_disease_detection = {0: 'Apple Black Rot Disease', 1: 'Apple Scab Disease', 2: 'Grape Esca (Black Measles) Disease', 3:'Grape Leaf Blight Disease',4:'Tomato Early Blight Disease',5:'Tomato Leaf Mold Disease'}
class_labels_of_water = {0: 'dead', 1: 'healthy', 2: 'unhealthy'}


# Load the classification model
model_path_classification = '../leaf_classification_model_1_0.h5'
model_classification = load_model(model_path_classification)
print(f"Model loaded successfully from {model_path_classification}")

# Load the disease detection model
model_path_disease = '../disease_detection_model_1_1.h5'
model_disease = load_model(model_path_disease)
print(f"Model loaded successfully from {model_path_disease}")

# Load the water model
model_path_water = '../water_level_model_1_0.h5'
model_water = load_model(model_path_water)
print(f"Model loaded successfully from {model_path_water}")


# ======================  Leaf Classification    ===========================
def preprocess_image_classification(img):
    img = img.resize((500, 500))  # Resize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values
    return img_array

def predict_image_classification(img, model_classification):
    preprocessed_image = preprocess_image_classification(img)
    prediction = model_classification.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    predicted_class = class_labels_of_classification[class_index]
    confidence = prediction[0][class_index]
    return predicted_class, confidence


# ========================== Disease Detection ===============================
def preprocess_image_disease(img):
    img = img.resize((256, 256))  # Resize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values
    return img_array

def predict_image_disease(img, model_disease):
    preprocessed_image = preprocess_image_disease(img)
    prediction = model_disease.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    predicted_class = class_labels_of_disease_detection[class_index]
    confidence = prediction[0][class_index]
    return predicted_class, confidence

# ========================== Water Estimation ===============================
def preprocess_image_water(img):
    img = img.resize((500, 500))  # Resize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values
    return img_array

def predict_image_water(img, model_water):
    preprocessed_image = preprocess_image_water(img)
    prediction = model_water.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    predicted_class = class_labels_of_water[class_index]
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
        predicted_class, confidence = predict_image_classification(img, model_classification)

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





@app.route('/disease_detect', methods=['POST'])
def disease_detect():
    try:
        # Check if the 'image' key is present in request.files
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the file from request.files
        file = request.files['image']

        # Open the image using PIL
        img = Image.open(io.BytesIO(file.read()))

        # Perform prediction
        predicted_class, confidence = predict_image_disease(img, model_disease)

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



@app.route('/water_estimation', methods=['POST'])
def water_estimation():
    try:
        # Check if the 'image' key is present in request.files
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the file from request.files
        file = request.files['image']

        # Open the image using PIL
        img = Image.open(io.BytesIO(file.read()))

        # Perform prediction
        predicted_class, confidence = predict_image_water(img, model_water)

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



# if __name__ == '__main__':
#     app.run(debug=True)
