import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
@st.cache_resource
def load_my_model():
    return load_model("final_model.keras")

model = load_my_model()

# Class labels (MUST match training order)
classes = ['Hatchback', 'Other', 'Pickup', 'SUV', 'Sedan']

st.title("Vehicle Type Classifier 🚗 by Imane Belbachir")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # open and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64, 64))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")