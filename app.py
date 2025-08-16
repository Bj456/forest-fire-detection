import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="🌲🌲 Innovative Forest Fire Detection AI App 🌲🌲",
    layout="centered"
)

# ----------------------
# Banner
# ----------------------
BANNER_PATH = "fire_banner.png"  # Replace with your banner image path
if os.path.exists(BANNER_PATH):
    banner = Image.open(BANNER_PATH)
    width_percent = 0.95  # Adjust width as % of container
    banner_width = int(banner.width * width_percent)
    banner_ratio = banner.width / banner.height
    banner_height = int(banner_width / banner_ratio)
    banner = banner.resize((banner_width, banner_height), Image.LANCZOS)
    st.image(banner, use_container_width=True)

# ----------------------
# Title and Subtitle (HTML)
# ----------------------
st.markdown("""
<h1 style='text-align: center; font-size: 42px;'>
🌲🌲 Innovative Forest Fire Detection AI App 🌲🌲
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size: 18px;'>
यह आर्टिफ़िशियल इंटेलिजेंस एप्लीकेशन जंगलों को आग से बचाने के लिए एक शैक्षिक नवाचार के रूप में योगदान है। - शिक्षाक नवाचारी भास्कर जोशी
</p>
""", unsafe_allow_html=True)

# ----------------------
# Load Model
# ----------------------
MODEL_PATH = "forest_fire_final_fixed.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please check the path!")
else:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")

# ----------------------
# Class mapping
# ----------------------
class_indices = {'Smoke': 0, 'fire': 1, 'non fire': 2}
idx_to_class = {v: k for k, v in class_indices.items()}

# ----------------------
# Prediction function
# ----------------------
def predict_image(img):
    img = img.resize((128,128))
    img_array = image.img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_probs = model.predict(img_array)
    pred_idx = np.argmax(pred_probs)
    pred_class = idx_to_class[pred_idx]
    confidence = pred_probs[0][pred_idx] * 100
    return pred_class, confidence, pred_probs[0]

# ----------------------
# Image Upload Prediction
# ----------------------
st.header("📁 Upload Image for Prediction")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    pred_class, confidence, all_probs = predict_image(img)

    emoji_dict = {'fire':'🔥','Smoke':'💨','non fire':'🌳'}
    st.subheader(f"Prediction: **{pred_class}** {emoji_dict[pred_class]}")

    st.subheader("Prediction Confidence")
    st.progress(int(confidence))
    if confidence > 80:
        st.success(f"Confidence: {confidence:.2f}%")
    elif confidence > 50:
        st.warning(f"Confidence: {confidence:.2f}%")
    else:
        st.error(f"Confidence: {confidence:.2f}%")

    prob_df = pd.DataFrame({
        'Class':['Smoke','Fire','Non-fire'],
        'Probability': all_probs*100
    })
    color_scale = alt.Scale(domain=['Smoke','Fire','Non-fire'], range=['#555555','red','green'])
    chart = alt.Chart(prob_df).mark_bar().encode(
        x='Class',
        y='Probability',
        color=alt.Color('Class', scale=color_scale)
    ).properties(width=400)
    st.altair_chart(chart)

    # Alerts + GIFs
    if pred_class=='fire' and confidence>80:
        st.balloons()
        st.markdown("⚠️ **आग लगी है! कृपया तुरंत आवश्यक कदम उठाएँ! 🔥🔥**")
        fire_gif = "fire_alert.gif"
        if os.path.exists(fire_gif):
            st.image(fire_gif)
    elif pred_class=='Smoke' and confidence>80:
        st.warning("💨 **धुआँ detected! शायद आग लग सकती है, कृपया आवश्यक सावधानी बरतें।**")
        smoke_gif = "smoke_alert.gif"
        if os.path.exists(smoke_gif):
            st.image(smoke_gif)
    else:
        st.success("🌳 **घबराए नहीं! कोई आग नहीं लगी है, आप निश्चित रहें। ✅**")
        safe_gif = "safe.gif"
        if os.path.exists(safe_gif):
            st.image(safe_gif)

    # Highlighted JSON-style probabilities
    prob_html = f"""
    <div style='background-color:#d4f4dd; padding:15px; border-radius:10px; font-family:monospace; font-size:18px;'>
    <pre style='margin:0;'>
{{
"Smoke": "<b style='color:#555555'>{all_probs[0]*100:.2f}%</b>",
"Fire": "<b style='color:red'>{all_probs[1]*100:.2f}%</b>",
"Non-fire": "<b style='color:green'>{all_probs[2]*100:.2f}%</b>"
}}
    </pre>
    </div>
    """
    st.markdown(prob_html, unsafe_allow_html=True)

# ----------------------
# Mobile/PC Webcam Prediction
# ----------------------
st.header("📷 Live Camera Prediction")

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

webrtc_streamer(
    key="webcam",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# ----------------------
# Footer
# ----------------------
st.markdown("""
<p style='text-align:center; font-size:14px; margin-top:30px;'>
AI Model शिक्षाक भास्कर जोशी द्वारा प्रशिक्षित किया गया है। सभी अधिकार सुरक्षित। License: CC BY
</p>
""", unsafe_allow_html=True)
