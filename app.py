import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load the pre-trained model
model = load_model(r"C:\Users\Ranjan kumar pradhan\.vscode\forest_fire_detection\Fire_Detection_model.h5")  # Ensure the model file is in the correct directory

# Mapping the predicted classes to labels
class_labels = {0: "Smoke", 1: "Fire", 2: "Non-Fire"}

# Function to preprocess and predict the image class
def predict_image(image_path):
    # Load the image with target size (150, 150)
    image = load_img(image_path, target_size=(150,150))
    # Convert the image to a NumPy array
    image_array = img_to_array(image)
    # Normalize the pixel values to [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension (1, img_width, img_height, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make predictions using the model
    predictions = model.predict(image_array)
    
    # Get the predicted class index with the highest probability
    predicted_class_index = np.argmax(predictions)
    
    # Map the index to a label
    predicted_label = class_labels[predicted_class_index]
    
    return predicted_label

# Streamlit UI
st.title("Forest Fire Detection:-")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    image = load_img(uploaded_file, target_size=(150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict the image
    st.write("Classifying the image as per model.")
    predicted_label = predict_image(uploaded_file)
    
    # Display the result
    st.write(f"Detection: The image is detected as:- **{predicted_label}**")

