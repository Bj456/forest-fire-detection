import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------
# Page Config
# ----------------------
st.set_page_config(page_title="📷 Forest Fire Camera App", layout="centered")

st.title("📷 Forest Fire Detection - Live Camera")

# ----------------------
# Load Model
# ----------------------
MODEL_PATH = "forest_fire_final_fixed.h5"
model = load_model(MODEL_PATH)

# ----------------------
# Class Mapping
# ----------------------
class_indices = {'Smoke': 0, 'fire': 1, 'non fire': 2}
idx_to_class = {v: k for k, v in class_indices.items()}

# ----------------------
# Video Transformer
# ----------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.idx_to_class = idx_to_class

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (128,128))
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred_probs = self.model.predict(img_array)
        pred_idx = np.argmax(pred_probs)
        pred_class = self.idx_to_class[pred_idx]
        confidence = pred_probs[0][pred_idx] * 100

        text = f"{pred_class} ({confidence:.1f}%)"
        color = (0,255,0) if pred_class=='non fire' else (0,0,255) if pred_class=='fire' else (200,200,200)
        cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# ----------------------
# Camera Selection
# ----------------------
camera_type = st.selectbox("📌 Select Camera", ["Front Camera", "Back Camera"])

constraints = {
    "video": {
        "facingMode": "user" if camera_type=="Front Camera" else {"exact":"environment"}
    },
    "audio": False
}

# ----------------------
# Start WebRTC
# ----------------------
webrtc_streamer(
    key="fire-camera",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints=constraints,
)

st.markdown(
    "<p style='text-align:center; font-size:14px; margin-top:20px;'>"
    "🚨 Use HTTPS link (e.g., Streamlit Cloud / ngrok) for mobile camera access"
    "</p>",
    unsafe_allow_html=True
)
