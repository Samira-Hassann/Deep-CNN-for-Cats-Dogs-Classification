import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="AI Vision | Cats & Dogs Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAppHeader {
        background-color: rgba(0,0,0,0);
    }
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        margin-top: 1rem;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #475569;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/dog--v1.png", width=100)
    st.title("🐾 AI Pet Vision")
    st.write("An AI-powered web app to classify images using Deep Convolutional Neural Networks (CNN).")
    st.divider()
    
    st.subheader("💡 Instructions")
    st.markdown("""
    1. Upload an image of a cat or a dog (`JPG` or `PNG`).
    2. Wait a moment for automatic processing.
    3. View the prediction result and model confidence score.
    """)
    st.divider()
    st.caption("Powered by TensorFlow & Streamlit")

# 4. Main Header
st.markdown('<h1 class="title-text">🐱🐶 Cats vs Dogs Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload an image for instant deep learning classification</p>', unsafe_allow_html=True)

# 5. Load Model with Caching & Verification
MODEL_NAME = 'best_cats_dogs_model (2).keras'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_NAME):
        st.error(f"❌ Model file `{MODEL_NAME}` not found. Make sure it is uploaded in the root directory on GitHub.")
        st.stop()
    return tf.keras.models.load_model(MODEL_NAME, safe_mode=False)

with st.spinner('Loading model...'):
    model = load_model()

# 6. Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📊 Analysis Result")
    
    if uploaded_file is not None:
        with st.spinner('Analyzing image...'):
            image_rgb = image.convert('RGB')
            size = (224, 224)
            image_resized = ImageOps.fit(image_rgb, size, Image.Resampling.LANCZOS)
            
            img_array = np.asarray(image_resized) / 255.0
            img_reshape = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_reshape)[0][0]
            
            if prediction > 0.5:
                score = prediction * 100
                st.balloons()
                st.markdown(f"""
                <div class="prediction-card" style="border-left: 6px solid #22c55e;">
                    <h2 style="color: #15803d; margin:0;">🐶 Prediction: Dog</h2>
                    <p class="confidence-text" style="margin-top:10px;">Confidence Score: <b>{score:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(score))
            else:
                score = (1 - prediction) * 100
                st.snow()
                st.markdown(f"""
                <div class="prediction-card" style="border-left: 6px solid #06b6d4;">
                    <h2 style="color: #0e7490; margin:0;">🐱 Prediction: Cat</h2>
                    <p class="confidence-text" style="margin-top:10px;">Confidence Score: <b>{score:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(score))
    else:
        st.info("👈 Please upload an image from the left side to start analysis.")
