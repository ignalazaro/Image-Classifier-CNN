import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gradio as gr

# Cargar el modelo preentrenado
model = tf.keras.models.load_model('imageclassifier.h5')

# Función para preprocesar la imagen antes de la predicción
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar
    return img_array

# Función para realizar la predicción
def predict_emotion(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction[0][0]

# Configurar la interfaz de usuario con Streamlit
st.title("Detector de Emociones")
st.write("Carga una imagen y predice si la cara está feliz o triste.")

# Sección para cargar la imagen
uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

    # Botón para realizar la predicción
    if st.button("Predecir Emoción"):
        # Realizar la predicción
        prediction = predict_emotion(uploaded_file)
        
        # Mostrar el resultado
        if prediction > 0.5:
            st.success("¡La cara parece estar feliz!")
            st.write("Probabilidad: ", prediction)
        else:
            st.error("¡La cara parece estar triste!")
            st.write("Probabilidad: ", 1- prediction)

# Configurar la interfaz de Gradio
iface = gr.Interface(fn=predict_emotion, 
                     inputs=gr.Image(shape=(256, 256)), 
                     outputs=gr.Label(num_top_classes=2))

# Mostrar la interfaz de Gradio
with st.echo(code_location='above'):
    iface.launch()
