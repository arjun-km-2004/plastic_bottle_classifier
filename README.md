# ♻️ Plastic Bottle Classifier (Recyclable vs Non-Recyclable)

This is a deep learning-powered plastic bottle classifier built with **EfficientNetV2S**. It distinguishes between **Recyclable**, **Non-Recyclable**, and **Uncertain** plastic waste using both `.h5` and `.tflite` models.

---

## 🚀 Features

- 🧠 EfficientNetV2S (with ImageNet weights)
- ⚖️ Binary Focal Loss (handles class imbalance)
- 🔁 Two-phase training (transfer learning + fine-tuning)
- 🧪 Evaluation: Accuracy plots, Confusion Matrix, Classification Report
- 🌐 Streamlit Web App (runs locally or on [Streamlit Cloud](https://streamlit.io/cloud))
- 📱 Compatible with MIT App Inventor or other mobile frontends
- 💾 Exports to `.h5` and `.tflite`

---

## 🖼️ Sample Images

Upload an image of a plastic bottle and the model will classify it into:

- ♻️ Recyclable
- 🚫 Non-Recyclable
- ⚠️ General Waste / Uncertain

---

## 🧪 Local Usage

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
__________________________________________________________________________________________________

# ♻️ Plastic Bottle Classifier

This is a **Streamlit app** that predicts whether a plastic bottle image is **Recyclable**, **Non-Recyclable**, or **General Waste / Uncertain** using a custom trained **EfficientNetV2S** model.

---

## 🗂️ How it works
- Upload an image (`.jpg`, `.jpeg`, `.png`)
- The app runs **both** a `.h5` Keras model **and** a `.tflite` TensorFlow Lite model
- Predictions are averaged for better stability
- Logic:
  - `> 0.6` → ♻️ **Recyclable**
  - `< 0.4` → 🚫 **Non-Recyclable**
  - Between 0.4–0.6 → ⚠️ **Uncertain / General Waste**

---

## 📦 Project structure

```plaintext
plastic_bottle_classifier/
├── models/
│   ├── efficient_bottle_classifier.h5
│   └── efficient_bottle_classifier.tflite
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
