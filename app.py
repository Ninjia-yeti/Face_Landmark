import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Set Streamlit app title
st.title('🦾 顔のランドマーク検出（画像のみ）')

# Load YOLOv8 model (use your trained weights)
model_path = 'best.pt'  # Path to your YOLOv8 trained weights
try:
    model = YOLO(model_path)  # Load the trained YOLOv8 pose model
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# File uploader for image files (JPG, JPEG, PNG)
uploaded_file = st.file_uploader('📤 画像のアップロード', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.subheader('アップロードされた画像:')

    # Open the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_container_width=True)
    except Exception as e:
        st.error(f"画像の処理に失敗しました: {e}")
        st.stop()

    if st.button('🔍 画像に対する検出の実行'):
        with st.spinner('顔のランドマークを検出しています...'):
            try:
                # Save the uploaded image temporarily
                image_path = './temp_uploaded_image.png'
                image.save(image_path)

                # Run YOLOv8 pose detection
                results = model.predict(source=image_path, save=True, conf=0.5)  # Confidence threshold at 50%

                # Get the path to the YOLOv8 result image
                result_image_path = results[0].save()

                # Display the detected image
                st.subheader('📸 検知結果:')
                st.image(result_image_path, caption='検出された顔のランドマーク', use_container_width=True)

                # Option to download the result image
                with open(result_image_path, 'rb') as file:
                    st.download_button(
                        label='💾 結果画像をダウンロード',
                        data=file,
                        file_name='detection_result.png',
                        mime='image/png'
                    )
            except Exception as e:
                st.error(f"検出処理中にエラーが発生しました: {e}")
