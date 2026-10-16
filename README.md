# 🐱🐶🐾 Deep CNN Model for Cats & Dogs Classification

An end-to-end Deep Learning project that classifies images of cats and dogs using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras. This project covers everything from rigorous Exploratory Data Analysis (EDA) and data cleaning to model training, evaluation, and interactive web deployment using Streamlit.

---

## 🔗 Project Links & Live Demos
* **Kaggle Notebook:** [View Notebook](https://www.kaggle.com/code/samoura/deep-cnn-model-for-cats-dogs-classification)
* **Live Web App:** [Try Streamlit App](https://cats-dogs-classification.streamlit.app/)

---

## 🚀 Key Features & Pipeline Steps

1. **Environment Setup & Data Loading:**
   * Utilized KaggleHub to download the dataset.
   * Loaded and structured image paths using Pandas and NumPy.

2. **Exploratory Data Analysis (EDA) & Data Quality Checks:**
   * **Class Balance Check:** Ensured a perfectly balanced dataset between cats and dogs.
   * **Data Integrity & Consistency:** Checked for corrupted files, analyzed image dimensions, fixed mixed color modes (RGB, RGBA, Grayscale), and inspected aspect ratios.
   * **Quality Filtering:** Removed extremely low-resolution images and duplicate entries using perceptual hashing.

3. **Data Preprocessing & Augmentation:**
   * Standardized image sizes to $224 \times 224$ pixels.
   * Applied data augmentation techniques (Rotation, Width/Height Shift, Shear, Zoom, Horizontal Flip, Brightness adjustment) to enhance model generalization.
   * Used `ImageDataGenerator` for efficient batch loading.

4. **Deep Learning Model Architecture:**
   * Built a custom sequential CNN consisting of multiple convolutional blocks, Batch Normalization, Max Pooling, and Dropout layers.
   * Concluded with Global Average Pooling and Dense layers with a Sigmoid activation function for binary classification.

5. **Callbacks & Training Strategy:**
   * **EarlyStopping:** Monitored validation loss to prevent overfitting.
   * **ModelCheckpoint:** Saved the best-performing model weights (`best_cats_dogs_model.keras`).
   * **ReduceLROnPlateau:** Dynamically adjusted the learning rate during training.

6. **Model Evaluation:**
   * Achieved an impressive **96.77% Test Accuracy**.
   * Evaluated performance using Confusion Matrices, Classification Reports (Precision, Recall, F1-Score), ROC Curves, and Precision-Recall Curves.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Image Processing:** OpenCV, PIL (Pillow)
* **Model Evaluation:** Scikit-learn
* **Deployment:** Streamlit

---


## 📁 Repository Structure

```text
.
├── app.py                         # Main Streamlit web application script
├── best_cats_dogs_model.keras     # Trained TensorFlow/Keras CNN model
├── requirements.txt               # Required Python packages for deployment
└── README.md                      # Project documentation
