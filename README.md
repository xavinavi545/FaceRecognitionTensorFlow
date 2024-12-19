# FaceRecognitionTensorFlow
# Proyecto: Reconocimiento Facial con TensorFlow y Transfer Learning

Este proyecto realiza **reconocimiento facial** usando TensorFlow con **Transfer Learning** (MobileNetV2) y organiza todo en pasos simples. El modelo se entrena para identificar **tu cara** frente a **otras caras** usando imágenes locales y un dataset preexistente.

---

## **Requisitos Previos**

Antes de comenzar, asegúrate de tener lo siguiente:
1. **Python 3.8+** instalado.
2. **pip** actualizado:

   ```bash
   python -m pip install --upgrade pip

FaceRecognitionTensorFlow/
├── dataset_faces/
│   ├── tu_cara/          # Aquí se guardarán tus fotos tomadas con la cámara
│   └── otras_caras/      # Fotos descargadas automáticamente (otras personas)
│
├── capture_photos.py       # Script para capturar tus fotos con la cámara
├── create_image_folders.py # Script para descargar imágenes de 'otras_caras'
├── train_face_recognition.py # Entrenamiento del modelo
├── predict_face.py         # Predicción con el modelo entrenado
├── requirements.txt        # Dependencias necesarias
└── README.md               # Este archivo con instrucciones paso a paso

Pasos para Hacer el Proyecto
1. Crear la Carpeta del Proyecto
 ```bash
  mkdir FaceRecognitionTensorFlow
  cd FaceRecognitionTensorFlow
 ```
Configurar el Entorno Virtual
python -m venv venv
venv\Scripts\activate
# Instalar las Dependencias
tensorflow
opencv-python
kagglehub
numpy
matplotlib

pip install -r requirements.txt


# Capturar Fotos para tu_cara
python capture_photos.py
Espacio: Captura una foto.
q: Salir del programa.

Las imágenes se guardarán en
dataset_faces/tu_cara/
Descargar Imágenes para otras_caras
Ejecuta el script create_image_folders.py para descargar imágenes de otras personas automáticamente:
python create_image_folders.py
# Entrenar el Modelo
python train_face_recognition.py
El modelo se entrenará y se guardará como face_recognition_model.h5.

# Probar el Modelo con una Imagen Nueva
python predict_face.py





