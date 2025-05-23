
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# بارگذاری مدل TFLite
interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
interpreter.allocate_tensors()

# دریافت اطلاعات ورودی و خروجی مدل
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("تشخیص پنومونی با هوش مصنوعی")
st.write("این اپلیکیشن با استفاده از مدل سبک‌شده TFLite، پنومونی را از روی عکس CXR تشخیص می‌دهد.")

uploaded_file = st.file_uploader("لطفاً یک عکس CXR آپلود کنید", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="تصویر ورودی", use_column_width=True)

    # پیش‌پردازش
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # اجرای مدل
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    class_names = ["Normal", "Pneumonia"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"**نتیجه پیش‌بینی:** {predicted_class}")
    st.markdown(f"**میزان اطمینان:** {confidence:.2f}%")
