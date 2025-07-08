import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from utils import preprocess_image  # Ensure preprocess_image is defined correctly

# Load the model
def load_trained_model(model_path):
    model = load_model(model_path, compile=False)
    return model

custom_objects = {
    'DepthwiseConv2D': DepthwiseConv2D
}

# Load model
model = load_trained_model('model/keras_model.h5')
class_names = ['Mask Detected', 'Mask Not Detected']

# Streamlit UI
st.title("😷 Mask Detection System")

menu = ['Upload Image', 'Live Camera']
choice = st.sidebar.selectbox('Select Mode', menu)

# Prediction function
def predict_image(image_input):
    img_array = preprocess_image(image_input)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    if predicted_class == 'Mask Detected':
        message = "✅ Mask is detected."
    else:
        message = "❌ No mask detected."

    return predicted_class, confidence, message

# Upload Image Mode
if choice == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True, width=300)
        predicted_class, confidence, message = predict_image(uploaded_file)
        st.success(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
        st.write(message)

# Live Camera Mode (using st.camera_input)
elif choice == 'Live Camera':
    st.warning("""
    📸 Please allow camera access when prompted by your browser.
    This feature works best in Chrome or Firefox.
    """)

    picture = st.camera_input("Take a picture")

    if picture:
        # Convert image to OpenCV format
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Captured Image", use_container_width=True)

        predicted_class, confidence, message = predict_image(picture)
        st.success(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
        st.write(message)
