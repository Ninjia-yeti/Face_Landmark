import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Set Streamlit app title
st.title('🦾 Facial Landmark Detection (Image Only)')

# Load YOLOv8 model (use your trained weights)
model_path = 'best.pt'  # Path to your YOLOv8 trained weights
model = YOLO(model_path)  # Load the trained YOLOv8 pose model

# File uploader for image files (JPG, JPEG, PNG)
uploaded_file = st.file_uploader('📤 Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.subheader('Uploaded Image:')

    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button('🔍 Run Detection on Image'):
        with st.spinner('Detecting facial landmarks...'):
            # Save the uploaded image temporarily
            image_path = './temp_uploaded_image.png'
            image.save(image_path)

            # Run YOLOv8 pose detection
            results = model.predict(source=image_path, save=True, conf=0.5)  # Confidence threshold at 50%

            # Get the exact path to the YOLOv8 result image
            result_image_path = results[0].save()  # Get the path to the saved YOLOv8 result

            # Display the detected image
            st.subheader('📸 Detection Result:')
            st.image(result_image_path, caption='Detected Facial Landmarks', use_container_width=True)

            # Option to download the result image
            with open(result_image_path, 'rb') as file:
                st.download_button(
                    label='💾 Download Result Image',
                    data=file,
                    file_name='detection_result.png',
                    mime='image/png'
                )
