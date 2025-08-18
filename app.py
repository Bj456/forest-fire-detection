import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="🌲 Forest Fire Detection AI App",
    layout="centered"
)

# ----------------------
# Title
# ----------------------
st.markdown("<h1 style='text-align: center; font-size: 48px;'>🌲 Forest Fire Detection AI App 🌲</h1>", unsafe_allow_html=True)

# ----------------------
# Load Model
# ----------------------
MODEL_PATH = "forest_fire_final_fixed.h5"
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

    # Emoji feedback
    emoji_dict = {'fire':'🔥', 'Smoke':'💨', 'non fire':'🌳'}
    st.subheader(f"Prediction: **{pred_class}** {emoji_dict[pred_class]}")

    # Confidence bar
    st.subheader("Prediction Confidence")
    st.progress(int(confidence))
    if confidence > 80:
        st.success(f"Confidence: {confidence:.2f}%")
    elif confidence > 50:
        st.warning(f"Confidence: {confidence:.2f}%")
    else:
        st.error(f"Confidence: {confidence:.2f}%")

    # Probabilities Bar Chart
    prob_df = pd.DataFrame({
        'Class': ['Smoke','Fire','Non-fire'],
        'Probability': all_probs * 100
    })
    color_scale = alt.Scale(domain=['Smoke','Fire','Non-fire'], range=['#555555','red','green'])
    chart = alt.Chart(prob_df).mark_bar().encode(
        x='Class',
        y='Probability',
        color=alt.Color('Class', scale=color_scale)
    ).properties(width=400)
    st.altair_chart(chart)

    # Alerts + GIFs
    if pred_class == 'fire' and confidence > 80:
        st.balloons()
        st.markdown("⚠️ **आग लगी है! कृपया तुरंत आवश्यक कदम उठाएँ! 🔥🔥**")
        fire_gif_path = "fire_alert.gif"
        if os.path.exists(fire_gif_path):
            st.image(fire_gif_path)
        # 🔹 Auto-play Siren using HTML audio
        st.markdown("""
        <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/emergency/emergency_siren_close_long.ogg" type="audio/ogg">
        </audio>
        """, unsafe_allow_html=True)

    elif pred_class == 'Smoke' and confidence > 80:
        st.warning("💨 **धुआँ detected! शायद आग लग सकती है, कृपया आवश्यक सावधानी बरतें।**")
        smoke_gif_path = "smoke_alert.gif"
        if os.path.exists(smoke_gif_path):
            st.image(smoke_gif_path)
    else:
        st.success("🌳 **घबराए नहीं! कोई आग नहीं लगी है, आप निश्चित रहें। ✅**")
        safe_gif_path = "safe.gif"
        if os.path.exists(safe_gif_path):
            st.image(safe_gif_path)

    # Highlighted JSON-style Probabilities
    st.subheader("All probabilities")
    prob_html = f"""
    <div style='background-color:#d4f4dd; padding:15px; border-radius:10px; font-family:monospace; font-size:18px;' >
    <pre style='margin:0;' >
{{
"Smoke": "<b style='color:#555555'>{all_probs[0]*100:.2f}%</b>",
"Fire": "<b style='color:red'>{all_probs[1]*100:.2f}%</b>",
"Non-fire": "<b style='color:green'>{all_probs[2]*100:.2f}%</b>"
}}
    </pre>
    </div>
    """
    st.markdown(prob_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='
position: fixed;
left: 0;
bottom: 0;
width: 100%;
display: flex;
justify-content: center;
align-items: center;
background-color:#f0f0f0;
padding:10px;
font-size:14px;
color:#333;
border-top:1px solid #ccc;
z-index:1000;
'>
AI Model शिक्षक भास्कर जोशी  द्वारा प्रशिक्षित। सभी अधिकार सुरक्षित। License: CC-BY-SA
</div>
""", unsafe_allow_html=True)
