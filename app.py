import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os

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
BANNER_PATH = "banner.jpg"  # Replace with your banner image path
if os.path.exists(BANNER_PATH):
    banner = Image.open(BANNER_PATH)
    st.image(banner, use_container_width=True)

# ----------------------
# Title and Subtitle (HTML)
# ----------------------
st.markdown("""
<h1 style='text-align: center; font-size: 48px;'>
🌲🌲 Innovative Forest Fire Detection AI App 🌲🌲
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size: 20px;'>
यह Artificial Intelligence एप्लीकेशन उत्तराखंड के नवाचारी शिक्षक भास्कर जोशी द्वारा उत्तराखंड के जंगलों को आग से बचाने के लिए एक नवाचार के रूप में योगदान है।
</p>
""", unsafe_allow_html=True)

# ----------------------
# Load Model
# ----------------------
MODEL_PATH = "forest_fire_final_fixed.h5"  # Updated model
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
# Image Upload
# ----------------------
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

# ----------------------
# Prediction
# ----------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    pred_class, confidence, all_probs = predict_image(img)

    # ----------------------
    # Emoji Feedback
    # ----------------------
    emoji_dict = {'fire':'🔥', 'Smoke':'💨', 'non fire':'🌳'}
    st.subheader(f"Prediction: **{pred_class}** {emoji_dict[pred_class]}")

    # ----------------------
    # Confidence Bar
    # ----------------------
    st.subheader("Prediction Confidence")
    st.progress(int(confidence))
    if confidence > 80:
        st.success(f"Confidence: {confidence:.2f}%")
    elif confidence > 50:
        st.warning(f"Confidence: {confidence:.2f}%")
    else:
        st.error(f"Confidence: {confidence:.2f}%")

    # ----------------------
    # Probabilities Bar Chart
    # ----------------------
    prob_df = pd.DataFrame({
        'Class': ['Smoke','Fire','Non-fire'],
        'Probability': all_probs * 100
    })

    chart = alt.Chart(prob_df).mark_bar().encode(
        x='Class',
        y='Probability',
        color='Class'
    ).properties(width=400)
    st.altair_chart(chart)

    # ----------------------
    # Interactive Hindi Alerts + GIFs
    # ----------------------
    if pred_class == 'fire' and confidence > 80:
        st.balloons()
        st.markdown("⚠️ **आग लगी है! कृपया तुरंत आवश्यक कदम उठाएँ! 🔥🔥**")
        fire_gif_path = "fire_alert.gif"
        if os.path.exists(fire_gif_path):
            st.image(fire_gif_path)
    elif pred_class == 'Smoke' and confidence > 80:
        st.warning("💨 **धुआँ detected! शायद आग लग सकती है, कृपया आवश्यक सावधानी बरतें।**")
        smoke_gif_path = "smoke_alert.gif"
        if os.path.exists(smoke_gif_path):
            st.image(smoke_gif_path)
    else:
        st.success("🌳 **गबराए नहीं! कोई आग नहीं लगी है, आप निश्चित रहें। ✅**")
        safe_gif_path = "safe.gif"
        if os.path.exists(safe_gif_path):
            st.image(safe_gif_path)

    # ----------------------
    # Highlighted Probabilities (Bold + Green)
    # ----------------------
    st.subheader("All probabilities (highlighted)")
    prob_html = f"""
    <div style='font-size:20px;'>
    <p><b style='color:#1f77b4'>Smoke:</b> <b style='color:green'>{all_probs[0]*100:.2f}%</b></p>
    <p><b style='color:#ff7f0e'>Fire:</b> <b style='color:green'>{all_probs[1]*100:.2f}%</b></p>
    <p><b style='color:#2ca02c'>Non-fire:</b> <b style='color:green'>{all_probs[2]*100:.2f}%</b></p>
    </div>
    """
    st.markdown(prob_html, unsafe_allow_html=True)
