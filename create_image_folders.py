import os
import kagglehub
import cv2

# Configuración
DATASET_OTRAS_CARAS = "jessicali9530/lfw-dataset"  # Dataset de Kaggle para descargar otras caras
BASE_DIR = "dataset_faces"  # Directorio principal donde se guardarán las imágenes

# Crear carpetas
def create_directories():
    os.makedirs(os.path.join(BASE_DIR, "tu_cara"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "otras_caras"), exist_ok=True)
    print("Carpetas creadas: 'tu_cara' (vacía) y 'otras_caras'.")

# Descargar imágenes del dataset usando KaggleHub
def download_dataset(dataset_name, save_path, max_images=50):
    print(f"Descargando imágenes del dataset: {dataset_name}")
    path = kagglehub.dataset_download(dataset_name)

    images_dir = os.path.join(path, "lfw-deepfunneled")  # Carpeta donde están las imágenes
    count = 0

    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".jpg") and count < max_images:
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                if img is not None:
                    save_file = os.path.join(save_path, f"image_{count}.jpg")
                    cv2.imwrite(save_file, img)
                    count += 1
    print(f"Imágenes descargadas: {count} en {save_path}")

# Ejecutar el script
if __name__ == "__main__":
    create_directories()
    
    # Descargar imágenes solo en 'otras_caras'
    print("Descargando imágenes para 'otras_caras'...")
    download_dataset(DATASET_OTRAS_CARAS, os.path.join(BASE_DIR, "otras_caras"), max_images=50)
