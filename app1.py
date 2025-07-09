import streamlit as st
import pickle
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #f0f9f0;
        }
        .title {
            font-size: 36px;
            font-weight: 600;
            color: #2e7d32;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 40px;
        }
        .prediction-box {
            background-color: #e8f5e9;
            padding: 16px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #1b5e20;
            margin-top: 20px;
        }
        .footer {
            font-size: 14px;
            color: #888;
            text-align: center;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_binarizer = pickle.load(open('label_binarizer.pkl', 'rb'))

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# ---------------------- Header ----------------------
st.markdown('<div class="title">🌿 Plant Leaf Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a plant leaf image to check its health status using a machine learning model trained on handcrafted features.</div>', unsafe_allow_html=True)

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2016/03/05/22/54/leaf-1231616_1280.jpg", use_container_width=True)
    st.write("This project uses HSV-based histogram features and a Random Forest model to detect plant leaf diseases. Built using Python, OpenCV, and Streamlit.")
    st.markdown("### 🔍 Classes Detected")
    st.write(", ".join(label_binarizer.classes_))
    st.markdown("---")
    st.caption("Developed by Shourya Saxena")

# ---------------------- File Upload ----------------------
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔬 Analyzing image..."):
        features = extract_features("temp.jpg")
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)

        # Convert to one-hot for LabelBinarizer
        num_classes = len(label_binarizer.classes_)
        one_hot = np.zeros((1, num_classes))
        one_hot[0, prediction[0]] = 1
        predicted_label = label_binarizer.inverse_transform(one_hot)[0]

    # Result Display
    st.markdown(f'<div class="prediction-box">🩺 Prediction: <span>{predicted_label}</span></div>', unsafe_allow_html=True)

    # Optional: Clean up temp file
    if os.path.exists("temp.jpg"):
        os.remove("temp.jpg")

# ---------------------- Footer ----------------------
st.markdown('<div class="footer">© 2025 Plant Pathology ML App | Built with Streamlit</div>', unsafe_allow_html=True)
