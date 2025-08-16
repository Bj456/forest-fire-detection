import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ------------------------------
# 🔹 Model load function
# ------------------------------
@st.cache_resource
def load_my_model():
    model = load_model("forest_fire_final.h5")
    return model

model = load_my_model()

# Class labels
class_names = ["fire", "non fire", "smoke"]
class_display = {
    "fire": "🔥 Fire detected",
    "smoke": "💨 Smoke detected",
    "non fire": "✅ No Fire"
}

# ------------------------------
# 🔹 Streamlit UI
# ------------------------------
st.title("🔥 Forest Fire Detection App")
st.write("Upload an image of a forest to check if it's fire, smoke, or safe.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_idx]
    confidence = prediction[predicted_idx] * 100  # percentage

    st.image(uploaded_file, caption=f"{class_display[predicted_class]} ({confidence:.2f}%)", use_container_width=True)
