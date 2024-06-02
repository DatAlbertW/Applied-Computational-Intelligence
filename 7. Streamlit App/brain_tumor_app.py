import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import gdown
import os

# Google Drive file ID
file_id = 'https://drive.google.com/file/d/1rXo1p7HeQlLWVH2JOq6ac2vevp6TuMA2/view?usp=sharing'
output = 'model_enb0.h5'

# Function to download the model from Google Drive
@st.cache(allow_output_mutation=True)
def download_model(file_id, output):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
    model = load_model(output)
    return model

# Load the model
model = download_model(file_id, output)

# Define class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.title("MRI Brain Tumor Diagnosis")

st.write("Upload an MRI image, and the model will diagnose the type of brain tumor or if there's no tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(224, 224))  # Adjust target_size according to your model input
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Create batch dimension

    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make predictions
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Output the diagnosis
    if predicted_class == 2:
        st.success("This is an MRI scan of a Healthy Patient")
    else:
        st.success(f"This is an MRI scan of a {class_names[predicted_class]}")

st.write("Note: The model is for educational purposes and not for medical diagnosis.")
