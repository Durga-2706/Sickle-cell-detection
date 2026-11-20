"""
Streamlit app: upload a red blood cell image and see the predicted class + confidence.

Run:  streamlit run app.py
EDUCATIONAL PROTOTYPE - NOT A MEDICAL DIAGNOSTIC TOOL.
"""
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sickle_cell_mobilenetv2.keras")

st.set_page_config(page_title="Sickle Cell Detection (Educational)", page_icon="🩸")
st.title("🩸 Sickle Cell Detection - Educational Prototype")
st.error("**Not a medical diagnostic tool.** This is a student project. "
         "Never use it to make or rule out a diagnosis. Consult a qualified professional.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def to_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_REPLICATE)


def preprocess(pil_img):
    # Same steps as training: pad to square -> resize 224x224 -> raw 0-255 RGB.
    # (Normalisation to [-1, 1] happens inside the model.)
    arr = np.array(pil_img.convert("RGB"))
    arr = cv2.resize(to_square(arr), (IMG, IMG))
    return np.expand_dims(arr.astype("float32"), 0)


if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Put `sickle_cell_mobilenetv2.keras` inside the `models/` folder.")
    st.stop()

st.write("Upload an image of a **single red blood cell** (a cropped cell, like the training images). "
         "Full blood-smear photos with many cells are not supported and will give meaningless results.")

file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", width=300)
    p_sickle = float(load_model().predict(preprocess(image), verbose=0)[0][0])
    label = "Sickle-shaped (elongated)" if p_sickle >= 0.5 else "Normal (circular)"
    confidence = p_sickle if p_sickle >= 0.5 else 1 - p_sickle
    st.subheader(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence * 100:.1f}%")
    st.caption("Confidence is the model's own probability estimate. It can be high and still wrong.")
