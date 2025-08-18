import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import tempfile
import threading
import os

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="🌲 Forest Fire Detection AI", layout="centered")

# ----------------------
# Title
# ----------------------
st.markdown("<h1 style='text-align:center;'>🌲 Forest Fire Detection AI 🌲</h1>", unsafe_allow_html=True)

# ----------------------
# Load model
# ----------------------
MODEL_PATH = "forest_fire_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}")
else:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")

input_height, input_width = model.input_shape[1], model.input_shape[2]
classes = ["No Fire", "Fire", "Smoke"]

# ----------------------
# Helper functions
# ----------------------
def preprocess(img):
    img = Image.fromarray(img).resize((input_width, input_height))
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)

def predict(img):
    processed = preprocess(img)
    preds = model.predict(processed, verbose=0)[0]
    pred_class = classes[np.argmax(preds)]
    confidence = preds[np.argmax(preds)] * 100
    return pred_class, confidence, preds

# ----------------------
# Auto siren
# ----------------------
def play_siren():
    st.markdown("""
    <audio autoplay>
      <source src="https://actions.google.com/sounds/v1/emergency/emergency_siren_close_long.ogg" type="audio/ogg">
    </audio>
    """, unsafe_allow_html=True)

# ----------------------
# Tabs: Upload & Webcam
# ----------------------
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        pred_class, confidence, all_probs = predict(np.array(img))

        emoji_dict = {'Fire':'🔥','Smoke':'💨','No Fire':'🌳'}
        st.subheader(f"Prediction: **{pred_class}** {emoji_dict[pred_class]}")
        st.progress(int(confidence))
        if confidence > 80:
            st.success(f"Confidence: {confidence:.2f}%")
        elif confidence > 50:
            st.warning(f"Confidence: {confidence:.2f}%")
        else:
            st.error(f"Confidence: {confidence:.2f}%")

        # GIF + Siren
        if pred_class == "Fire" and confidence > 80:
            play_siren()
            fire_gif_path = "fire_alert.gif"
            if os.path.exists(fire_gif_path):
                st.image(fire_gif_path)
            st.markdown("⚠️ **आग लगी है! कृपया तुरंत कदम उठाएँ! 🔥**")
        elif pred_class == "Smoke" and confidence > 80:
            smoke_gif_path = "smoke_alert.gif"
            if os.path.exists(smoke_gif_path):
                st.image(smoke_gif_path)
            st.warning("💨 **धुआँ detected! सावधानी बरतें।**")
        else:
            safe_gif_path = "safe.gif"
            if os.path.exists(safe_gif_path):
                st.image(safe_gif_path)
            st.success("🌳 **कोई आग नहीं है।**")

with tab2:
    st.info("Webcam detection running...")
    run_webcam = st.checkbox("Start Webcam Fire Detection")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not found!")
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_class, confidence, _ = predict(img_rgb)

            display_frame = img_rgb.copy()
            cv2.putText(display_frame, f"{pred_class} ({confidence:.1f}%)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            st_frame.image(display_frame)

            # Auto siren
            if pred_class=="Fire" and confidence > 80:
                play_siren()
