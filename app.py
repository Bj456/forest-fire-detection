import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("🌲 Forest Fire Detection App")

# 🔹 Load model
model = load_model("forest_fire_final.h5")

# Class mapping
class_indices = {'Smoke': 0, 'fire': 1, 'non fire': 2}
idx_to_class = {v: k for k, v in class_indices.items()}

# 🔹 Image upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = img.resize((128,128))
    img_array = image.img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred_probs = model.predict(img_array)
    pred_idx = np.argmax(pred_probs)
    pred_class = idx_to_class[pred_idx]
    confidence = pred_probs[0][pred_idx] * 100
    
    # Show result
    st.subheader("Prediction Result")
    st.write(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
