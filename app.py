import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="🌲 Forest Fire Detection", layout="centered")
st.title("🌲 Forest Fire Detection App")

# 🔹 Load model
MODEL_PATH = "forest_fire_final_fixed.h5"  # Updated model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please check the path!")
else:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")

# 🔹 Class mapping
class_indices = {'Smoke': 0, 'fire': 1, 'non fire': 2}
idx_to_class = {v: k for k, v in class_indices.items()}

# 🔹 Image upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

def predict_image(img):
    img = img.resize((128,128))
    img_array = image.img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_probs = model.predict(img_array)
    pred_idx = np.argmax(pred_probs)
    pred_class = idx_to_class[pred_idx]
    confidence = pred_probs[0][pred_idx] * 100
    return pred_class, confidence, pred_probs[0]

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # 🔹 Predict
    pred_class, confidence, all_probs = predict_image(img)

    # 🔹 Show result
    st.subheader("Prediction Result")
    st.write(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.write("All probabilities:")
    st.write({
        'Smoke': f"{all_probs[0]*100:.2f}%",
        'Fire': f"{all_probs[1]*100:.2f}%",
        'Non-fire': f"{all_probs[2]*100:.2f}%"
    })
