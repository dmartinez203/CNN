import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CNN Image Classifier Demo", layout="centered")

@st.cache_resource
def load_trained_model(path="best_cnn_cifar10.h5"):
    try:
        model = load_model(path)
    except Exception as e:
        st.error(f"Failed to load model at {path}: {e}")
        raise
    return model

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.title("CNN Image Classifier — CIFAR-10 demo")
st.write("Upload an image and the model will predict its CIFAR-10 class.")

model = None
try:
    model = load_trained_model()
except Exception:
    st.warning("Model could not be loaded. Make sure 'best_cnn_cifar10.h5' exists in the app folder.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Preprocess: resize to 32x32 and normalize
    img_resized = img.resize((32, 32))
    img_array = np.asarray(img_resized).astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # shape (1,32,32,3)

    if model is not None:
        with st.spinner("Predicting..."):
            preds = model.predict(img_batch)
            top_idx = np.argmax(preds[0])
            confidence = preds[0][top_idx]

        st.markdown(f"**Prediction:** {CLASS_NAMES[top_idx]}  \n**Confidence:** {confidence:.3f}")

        # Show full probabilities
        st.subheader("All class probabilities")
        prob_dict = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        st.table(prob_dict)
    else:
        st.error("Model not available. Please train the model in the notebook and ensure the checkpoint file 'best_cnn_cifar10.h5' is saved in this folder.")
