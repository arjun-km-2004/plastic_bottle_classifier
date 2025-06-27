import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np

# Load models
h5_model = load_model("models/efficient_bottle_classifier.h5", compile=False)

interpreter = tf.lite.Interpreter(model_path="models/efficient_bottle_classifier.tflite")
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
=======
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np

# Load models
h5_model = load_model("models/efficient_bottle_classifier.h5", compile=False)

interpreter = tf.lite.Interpreter(model_path="models/efficient_bottle_classifier.tflite")
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

