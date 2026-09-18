import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. إعداد الصفحة وتصميم الهيكل الرئيسي
st.set_page_config(
    page_title="AI Vision | Cats & Dogs Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. إضافة لمسات وتنسيقات CSS مخصصة لإعطاء مظهر احترافي
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
""", unsafe_unsafe_html=True)

# 3. الشريط الجانبي (Sidebar)
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/dog--v1.png", width=100)
    st.title("🐾 AI Pet Vision")
    st.write("تطبيق ذكاء اصطناعي لتصنيف الصور باستخدام شبكات التلافيف العميقة (CNN).")
    st.divider()
    
    st.subheader("💡 تعليمات الاستخدام")
    st.markdown("""
    1. ارفع صورة لقطة أو كلب بصيغة `JPG` أو `PNG`.
    2. انتظر ثانية لمعالجة الصورة تلقائياً.
    3. استعرض النتيجة ونسبة ثقة النموذج.
    """)
    st.divider()
    st.caption("Powered by TensorFlow & Streamlit")

# 4. العنوان الرئيسي للموقع
st.markdown('<h1 class="title-text">🐱🐶 Cats vs Dogs Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">رفع صورة للتعرف التلقائي عليها بواسطة نموذج الذكاء الاصطناعي</p>', unsafe_allow_html=True)

# 5. تحميل النموذج بالتخزين المؤقت
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_cats_dogs_model (2).keras')

with st.spinner('جاري تحميل النموذج...'):
    model = load_model()

# 6. تقسيم الواجهة إلى أعمدة (Columns) لتنظيم العرض
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 رفع الصورة")
    uploaded_file = st.file_uploader(
        "اختر صورة لرفعها...", 
        type=["jpg", "jpeg", "png"],
        help="تدعم الصيغ: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)

with col2:
    st.subheader("📊 نتيجة التحليل")
    
    if uploaded_file is not None:
        with st.spinner('جاري تحليل الصورة...'):
            # معالجة الصورة
            image_rgb = image.convert('RGB')
            size = (224, 224)
            image_resized = ImageOps.fit(image_rgb, size, Image.Resampling.LANCZOS)
            
            img_array = np.asarray(image_resized) / 255.0
            img_reshape = np.expand_dims(img_array, axis=0)
            
            # التنبؤ
            prediction = model.predict(img_reshape)[0][0]
            
            # عرض النتائج بشكل أنيق
            if prediction > 0.5:
                score = prediction * 100
                st.balloons()
                st.markdown(f"""
                <div class="prediction-card" style="border-left: 6px solid #22c55e;">
                    <h2 style="color: #15803d; margin:0;">🐶 التنبؤ: كلب (Dog)</h2>
                    <p class="confidence-text" style="margin-top:10px;">نسبة الثقة: <b>{score:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(score))
            else:
                score = (1 - prediction) * 100
                st.snow()
                st.markdown(f"""
                <div class="prediction-card" style="border-left: 6px solid #06b6d4;">
                    <h2 style="color: #0e7490; margin:0;">🐱 التنبؤ: قطة (Cat)</h2>
                    <p class="confidence-text" style="margin-top:10px;">نسبة الثقة: <b>{score:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(score))
    else:
        st.info("👈 قم برفع صورة من القائمة على اليسار لبدء التحليل.")
