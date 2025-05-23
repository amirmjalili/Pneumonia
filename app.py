
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# مدل را بارگذاری کن
model = tf.keras.models.load_model("pneumonia_detector_mobilenetv2_finetuned.h5")

st.title("تشخیص پنومونی از روی عکس CXR")

uploaded_file = st.file_uploader("یک تصویر CXR آپلود کنید", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='تصویر ورودی', use_column_width=True)

    # پیش‌پردازش تصویر
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # پیش‌بینی
    prediction = model.predict(img_array)
    class_names = ["Normal", "Pneumonia"]

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"**پیش‌بینی مدل:** {predicted_class}")
    st.markdown(f"**درصد اطمینان:** {confidence:.2f}%")
