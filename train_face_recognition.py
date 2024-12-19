import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# Configuración
DATASET_PATH = "dataset_faces"  # Ruta de las imágenes
IMAGE_SIZE = (128, 128)         # Tamaño de la imagen
BATCH_SIZE = 16
EPOCHS = 5                      # Número de épocas para entrenamiento
MODEL_PATH = "face_recognition_model.h5"

# Preprocesamiento de las imágenes
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Crear el modelo usando MobileNetV2 como base
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Congela las capas del modelo base

# Crear el modelo completo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Salida binaria
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
print("Entrenando el modelo...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Guardar el modelo
model.save(MODEL_PATH)
print(f"Modelo guardado como '{MODEL_PATH}'.")

# Mostrar gráfica de entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión del Modelo')
plt.show()
