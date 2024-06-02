import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def reassemble_model(parts, output_file):
    with open(output_file, 'wb') as model_file:
        for part in parts:
            with open(part, 'rb') as part_file:
                model_file.write(part_file.read())

# Reassemble the model if it doesn't exist
model_path = 'efficientnetb0_model.h5'
if not os.path.exists(model_path):
    model_parts = [
        '7. Streamlit App/efficientnetb0_model_part_aa',
        '7. Streamlit App/efficientnetb0_model_part_ab',
        '7. Streamlit App/efficientnetb0_model_part_ac',
    ]
    reassemble_model(model_parts, model_path)

# Load the reassembled model
@st.cache(allow_output_mutation=True)
def load_complete_model():
    model = load_model(model_path)
    return model

# Load the model
model = load_complete_model()

# Define class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# User Interface
st.title("MRI Brain Tumor Diagnosis")

def reset_state():
    st.session_state['uploaded_file'] = None
    st.session_state['prediction'] = None

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

st.write("Upload an MRI image, and the model will diagnose the type of brain tumor or if there's no tumor.")
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"], key="uploader")

if st.session_state['uploaded_file'] is None:
    st.session_state['uploaded_file'] = uploaded_file

if st.session_state['uploaded_file'] is not None:
    image = load_img(st.session_state['uploaded_file'], target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make predictions
    try:
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        if predicted_class == 2:
            st.session_state['prediction'] = "This is an MRI scan of a Healthy Patient"
        else:
            st.session_state['prediction'] = f"This is an MRI scan of a {class_names[predicted_class]}"
    except Exception as e:
        st.session_state['prediction'] = "Image not Recognized"

if st.session_state['prediction'] is not None:
    st.success(st.session_state['prediction'])

if st.button("Reset"):
    reset_state()
    st.experimental_rerun()

st.write("Note: The model is for educational purposes and not for medical diagnosis.")
