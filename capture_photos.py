import cv2
import os

# Configuración
SAVE_PATH = "dataset_faces/tu_cara"  # Carpeta donde se guardarán las fotos
os.makedirs(SAVE_PATH, exist_ok=True)  # Crear la carpeta si no existe

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Presiona 'ESPACIO' para tomar una foto, 'q' para salir...")

# Contador de imágenes
image_count = 0

while True:
    # Capturar el frame desde la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Mostrar la imagen en una ventana
    cv2.imshow("Captura de Fotos - Presiona ESPACIO", frame)

    # Escuchar las teclas presionadas
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Presiona ESPACIO para capturar una foto
        image_name = os.path.join(SAVE_PATH, f"tu_cara_{image_count}.jpg")
        cv2.imwrite(image_name, frame)
        print(f"Foto guardada: {image_name}")
        image_count += 1

    elif key == ord('q'):  # Presiona 'q' para salir
        print("Saliendo...")
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
