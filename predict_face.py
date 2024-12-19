import tensorflow as tf
import cv2
import numpy as np

# Ruta al modelo entrenado y la imagen de prueba
MODEL_PATH = "face_recognition_model.h5"
TEST_IMAGE_PATH = "C:\\Users\\Xavi\\Desktop\\FaceRecognitionTensorFlow\\FotopruebaXavier.jpeg"

# Cargar el modelo entrenado
print("Cargando el modelo entrenado...")
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar y preprocesar la imagen
print(f"Procesando la imagen: {TEST_IMAGE_PATH}")
image = cv2.imread(TEST_IMAGE_PATH)
image = cv2.resize(image, (128, 128))  # Redimensionar
image = image / 255.0  # Normalizar
image = np.expand_dims(image, axis=0)  # Añadir dimensión batch

# Realizar la predicción
prediction = model.predict(image)[0][0]
label = "tu_cara" if prediction > 0.5 else "otras_caras"
print(f"Predicción: {label} (Confianza: {prediction:.2f})")
