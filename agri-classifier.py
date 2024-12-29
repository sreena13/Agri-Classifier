import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os

image_size = (128, 128)

def predict_image(image_path, confidence_threshold=0.6):
    try:
        model, label_map = joblib.load('trained_model.pkl')
        image = Image.open(image_path).convert('RGB').resize(image_size)
        image_data = np.array(image).flatten() / 255.0
        prediction_probs = model.predict_proba([image_data])[0]
        max_confidence = max(prediction_probs)
        predicted_label_idx = np.argmax(prediction_probs)

        if max_confidence >= confidence_threshold:
            return label_map[predicted_label_idx], max_confidence
        else:
            return "Unknown", max_confidence
    except Exception as e:
        return f"Error: {e}", 0

# Streamlit app
st.title("Agricultural Crop Classification")
st.write("Upload an image to classify it into one of the predefined crop categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Save uploaded file temporarily
    temp_path = os.path.join("temp_image.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict
    predicted_label, confidence = predict_image(temp_path)

    if predicted_label != "Error":
        st.write(f"**Predicted Category:** {predicted_label}")
        st.write(f"**Confidence Score:** {confidence:.2f}")
    else:
        st.error("An error occurred during prediction.")

