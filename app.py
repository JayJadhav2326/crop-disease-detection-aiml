# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("AlexNetModel_trained.h5")
    return model

model = load_model()

# App Title
st.title("Crop Disease Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image of the crop leaf...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    st.write("Classifying...")
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    
    # Display result
    if class_idx == 0:
        st.write("The leaf is healthy.")
    elif class_idx == 1:
        st.write("The leaf has disease X.")
    elif class_idx == 2:
        st.write("The leaf has disease Y.")
    else:
        st.write("Unknown disease.")
