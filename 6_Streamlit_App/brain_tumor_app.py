import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
import io

# Path to the complete model
model_path = '6_Streamlit_App/copy_efficientnetb0_model.h5'

# Load the complete model
@st.cache(allow_output_mutation=True)
def load_complete_model():
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_complete_model()

# Check if the model was loaded successfully
if model is None:
    st.error("Failed to load the model. Please check the model path and try again.")
else:
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
        try:
            # Read the uploaded file
            file = st.session_state['uploaded_file']
            if file is not None:
                # Ensure the file pointer is at the start
                file.seek(0)
                # Use PIL to handle the uploaded file
                image = load_img(file, target_size=(224, 224))
                image_array = img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)
                image_array = preprocess_input(image_array)  # Ensure the image is preprocessed correctly

                st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
                st.write("")
                st.write("Classifying...")

                # Make predictions
                try:
                    predictions = model.predict(image_array)
                    st.write(f"Predicted probabilities: {predictions}")  # Debug: Show the predicted probabilities
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    st.session_state['prediction'] = f"This is an MRI scan of a {class_names[predicted_class]}"
                except Exception as e:
                    st.error(f"Error during model prediction: {e}")
                    st.session_state['prediction'] = "Image not Recognized"
            else:
                st.session_state['prediction'] = "No file uploaded or file could not be read."
        except Exception as e:
            st.error(f"Error during file processing: {e}")
            st.session_state['prediction'] = "Image not Recognized"

    if st.session_state['prediction'] is not None:
        st.success(st.session_state['prediction'])

    if st.button("Reset"):
        reset_state()
        st.experimental_rerun()

    st.write("Note: The model is for educational purposes and not for medical diagnosis.")

