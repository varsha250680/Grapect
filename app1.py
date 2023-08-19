from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Define the classes and their remedies
classes = {
    0: 'Grape Black Measles',
    1: 'Grape Black rot',
    2: 'Grape Healthy',
    3: 'Grape Isariopsis Leaf Spot'
}

# Define the remedies for each class
remedies = {
    0: ['Neem Oil', 'Baking Soda', 'Garlic Spray'],
    1: ['Apple Cider Vinegar','Milk Spray ',' Cinnamon Powder'],
    2: 'No remedy required.',
    3: ['Good Cultural Practices','Well-Rotted manure','Compost Tea']
}

# Set the input image size
input_size = (128, 128)

# Load the trained model
def load_model():
    model_path = 'trained_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Perform prediction on the image
def predict_image(image):
    # Preprocess the image
    image = image.convert('RGB')
    image = image.resize(input_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Perform prediction using the trained model
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index


# Define the home page route
@app.route('/')
def home():
    return render_template('index1.html')

# Define the result page route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the request
        image = request.files['image']

        # Open the image using PIL
        image = Image.open(image)

        # Perform prediction on the image
        predicted_class_index = predict_image(image)
        predicted_class = classes[predicted_class_index]
        remedy = remedies[predicted_class_index]

        # Render the result template with the predicted class and remedy
        return render_template('result.html', predicted_class=predicted_class, remedy=remedy)

if __name__ == '__main__':
    # Load the trained model
    model = load_model()

    # Run the Flask app
    app.run(debug=True)
