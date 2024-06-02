# App can be found in: https://applied-computational-intelligence-mag26btztvy8jjjnquqici.streamlit.app/
<![qr-code](https://github.com/DatAlbertW/Applied-Computational-Intelligence/assets/144963224/e4949176-6a01-40a4-947a-6bd05a21fcb6) width=30%>

# MRI Brain Tumor Diagnosis App

This Streamlit app uses a trained EfficientNetB0 model to diagnose the type of brain tumor from MRI images. The possible classifications are Glioma, Meningioma, Pituitary Tumor, or No Tumor.

## Features

- **Upload MRI Images:** Accepts JPG, JPEG, and PNG formats.
- **Predict Tumor Type:** Uses a pre-trained EfficientNetB0 model.
- **Display Results:** Shows the predicted classification and probability.

## Installation

1. Clone this repository.
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Usage

1. **Upload an Image:** Click "Choose an MRI image..." and select an MRI scan.
2. **View Prediction:** The app will display the uploaded image and the predicted tumor type.

## Model

The model file is located at `6_Streamlit_App/copy_efficientnetb0_model.h5`. Ensure this path is correct when running the app.

## Example Images

You can find example MRI images to test the app in the repository under `3. Software Prototype`.

## QR Code

Scan the QR code below to access the app:

<img src="path_to_qr_code_image" alt="QR Code" width="30%">

## Disclaimer

This app is for educational purposes only and is not intended for medical diagnosis.
