import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Cats vs Dogs Classifier 🐱🐶",
    page_icon="🐾",
    layout="centered"
)

# App Title & Description
st.title("🐱🐶 Image Classification: Cat or Dog?")
st.write("Upload an image of a cat or a dog, and the model will predict its class instantly.")

# Load model with caching for speed optimization
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_cats_dogs_model.keras')
    return model

with st.spinner('Loading model...'):
    model = load_model()

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image using updated Streamlit parameter
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Analyzing...")
    
    # Preprocess image (RGB conversion + Resize to 224x224)
    image = image.convert('RGB')
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize (1/255)
    img_array = np.asarray(image_resized) / 255.0
    
    # Add batch dimension -> Shape becomes (1, 224, 224, 3)
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction = model.predict(img_reshape)[0][0]
    
    # Display result based on Sigmoid output
    if prediction > 0.5:
        score = prediction * 100
        st.success(f"🐶 **Prediction: Dog** (Confidence: {score:.2f}%)")
    else:
        score = (1 - prediction) * 100
        st.success(f"🐱 **Prediction: Cat** (Confidence: {score:.2f}%)")
