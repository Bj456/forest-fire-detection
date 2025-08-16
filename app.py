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

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(uploaded_file, caption=f"Prediction: {predicted_class}", use_column_width=True)
    st.write("Prediction Probabilities:", prediction)
