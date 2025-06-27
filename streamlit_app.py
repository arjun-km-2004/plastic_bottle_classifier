import os
import requests
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np

# Model paths
MODEL_DIR = "models"
H5_MODEL_PATH = os.path.join(MODEL_DIR, "efficient_bottle_classifier.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "efficient_bottle_classifier.tflite")

# Make models folder
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive direct download links
H5_URL = "https://drive.google.com/uc?export=download&id=1ick5E5olrj9NTwY5MmW_U1DemR-CKKXA"
TFLITE_URL = "https://drive.google.com/uc?export=download&id=1K6a1duye0tM8kqfcDqcqG5ATNiBc8xef"

# Download .h5 if missing
if not os.path.exists(H5_MODEL_PATH):
    print("🔽 Downloading H5 model...")
    r = requests.get(H5_URL)
    with open(H5_MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ H5 model downloaded!")

# Download .tflite if missing
if not os.path.exists(TFLITE_MODEL_PATH):
    print("🔽 Downloading TFLite model...")
    r = requests.get(TFLITE_URL)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ TFLite model downloaded!")

# Load models
h5_model = load_model(H5_MODEL_PATH, compile=False)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess
def preprocess(img):
    img = img.resize((384, 384))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predictions
def predict_h5(img_array):
    return float(h5_model.predict(img_array)[0][0])

def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]['index'])[0][0])

# App UI
st.set_page_config(page_title="Plastic Bottle Classifier", layout="centered")
st.title("♻️ Plastic Bottle Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img_array = preprocess(image)
        pred_h5 = predict_h5(img_array)
        pred_tflite = predict_tflite(img_array)
        final_pred = (pred_h5 + pred_tflite) / 2

        # Logic
        if final_pred < 0 or final_pred > 1:
            label = "⚠️ General Waste / Uncertain"
            confidence = 0.0
        elif final_pred >= 0.6:
            label = "♻️ Recyclable"
            confidence = final_pred
        elif final_pred <= 0.4:
            label = "🚫 Non-Recyclable"
            confidence = 1 - final_pred
        else:
            label = "⚠️ General Waste / Uncertain"
            confidence = 1 - abs(0.5 - final_pred) * 2

        # Show result
        st.markdown(f"🔢 **Raw Prediction Score:** `{round(final_pred, 6)}`")
        st.markdown(f"🧠 **Prediction:** `{label}`")
        st.markdown(f"🔍 **Confidence:** `{round(confidence * 100, 2)}%`")
