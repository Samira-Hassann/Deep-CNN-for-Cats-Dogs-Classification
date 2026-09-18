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
    model = tf.keras.models.load_model('best_cats_dogs_model (2).keras')
    return model

with st.spinner('Loading model...'):
    model = load_model()

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Analyzing...")
    
    # Preprocess image matching model requirements (224x224 & 1/255 scaling)
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized)
    
    # Ensure RGB channels (handles RGBA PNG uploads)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
        
    img_reshape = img_array / 255.0  # Rescale
    img_reshape = np.expand_dims(img_reshape, axis=0)  # Add batch dimension
    
    # Prediction
    prediction = model.predict(img_reshape)[0][0]
    
    # Display result based on Sigmoid output
    if prediction > 0.5:
        score = prediction * 100
        st.success(f"🐶 **Prediction: Dog** (Confidence: {score:.2f}%)")
    else:
        score = (1 - prediction) * 100
        st.success(f"🐱 **Prediction: Cat** (Confidence: {score:.2f}%)")
