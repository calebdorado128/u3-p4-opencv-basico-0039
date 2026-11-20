# Caleb Dorado NC = 0039
# PRACTICA: OPERACIONES BASICAS CON OPENCV

# ==============================================================================
# 0. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import cv2  # Importa OpenCV para el procesamiento de imágenes.
import numpy as np  # Importa NumPy para trabajar con matrices.
import os  # Permite trabajar con rutas y archivos.

# ==============================================================================
# 1. LECTURA DE LA IMAGEN DESDE LA CARPETA
# ==============================================================================
print("\n1. LECTURA DE IMÁGENES")

ruta = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta, "token.jpeg")

IMG_TOKEN = cv2.imread(ruta_imagen)

if IMG_TOKEN is None:
    print("Error: No se pudo leer token.jpeg. Revisa el nombre y ubicación.")
    exit()
else:
    print("Imagen cargada exitosamente.")

# ==============================================================================
# 2. PROPIEDADES DE LA IMAGEN
# ==============================================================================
print("\n2. PROPIEDADES DE LA IMAGEN")

height, width, channels = IMG_TOKEN.shape

print(f"Dimensiones de la imagen: {width}x{height} píxeles")
print(f"Número de canales de color: {channels}")
print(f"Tipo de dato de la imagen: {IMG_TOKEN.dtype}")
print(f"Total de datos/píxeles: {IMG_TOKEN.size}")

# ==============================================================================
# 3. VISUALIZACIÓN DE LA IMAGEN ORIGINAL
# ==============================================================================
print("\n3. MOSTRANDO IMAGEN ORIGINAL")
print("Presiona cualquier tecla sobre la ventana para continuar...")

cv2.imshow("TOKEN Original 0039", IMG_TOKEN)
cv2.waitKey(0)

# ==============================================================================
# 4. ACCESO Y MODIFICACIÓN DE PÍXELES INDIVIDUALES
# ==============================================================================
print("\n4. ACCEDIENDO Y MODIFICANDO PÍXELES")

pixel_img = IMG_TOKEN.copy()

y, x = 100, 150

pixel = pixel_img[y, x]

print(f"Valor del píxel en ({x}, {y}): {pixel} (Formato BGR)")

pixel_img[y, x] = [0, 0, 255]

cv2.circle(pixel_img, (x, y), 15, (0, 255, 255), 2)

cv2.imshow("Pixel Modificado 0039", pixel_img)
cv2.waitKey(0)

# ==============================================================================
# 5. REGIÓN DE INTERÉS (ROI - REGION OF INTEREST)
# ==============================================================================
print("\n5. REGIÓN DE INTERÉS (ROI)")

roi_img = IMG_TOKEN.copy()

roi = roi_img[50:200, 100:300]

cv2.rectangle(roi_img, (100, 50), (300, 200), (0, 255, 0), 2)

cv2.imshow("Imagen con ROI marcada 0039", roi_img)
cv2.waitKey(0)

cv2.imshow("ROI Extraida 0039", roi)
cv2.waitKey(0)

# ==============================================================================
# 6. SEPARACIÓN Y FUSIÓN DE CANALES DE COLOR (BGR)
# ==============================================================================
print("\n6. SEPARANDO Y FUSIONANDO CANALES")

b, g, r = cv2.split(IMG_TOKEN)

cv2.imshow("Canal Azul 0039", b)
cv2.imshow("Canal Verde 0039", g)
cv2.imshow("Canal Rojo 0039", r)
cv2.waitKey(0)

zeros = np.zeros_like(b)

blue_img = cv2.merge([b, zeros, zeros])
green_img = cv2.merge([zeros, g, zeros])
red_img = cv2.merge([zeros, zeros, r])

cv2.imshow("Canal Azul Coloreado 0039", blue_img)
cv2.imshow("Canal Verde Coloreado 0039", green_img)
cv2.imshow("Canal Rojo Coloreado 0039", red_img)
cv2.waitKey(0)

# ==============================================================================
# 7. REDIMENSIONAMIENTO DE IMÁGENES (RESIZING)
# ==============================================================================
print("\n7. REDIMENSIONANDO IMÁGENES")

resized_img = cv2.resize(IMG_TOKEN, (400, 300))

cv2.imshow("Redimensionada 400x300 0039", resized_img)
cv2.waitKey(0)

half_size = cv2.resize(IMG_TOKEN, None, fx=0.5, fy=0.5)

double_size = cv2.resize(IMG_TOKEN, None, fx=1.5, fy=1.5)

cv2.imshow("Mitad de tamano 0039", half_size)
cv2.waitKey(0)

cv2.imshow("Ampliada 1.5x 0039", double_size)
cv2.waitKey(0)

# ==============================================================================
# 8. ROTACIÓN DE IMÁGENES
# ==============================================================================
print("\n8. ROTANDO IMÁGENES")

height, width = IMG_TOKEN.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D(
    (width / 2, height / 2), 45, 1
)

rotated_img = cv2.warpAffine(
    IMG_TOKEN, rotation_matrix, (width, height)
)

cv2.imshow("Imagen Rotada 45 grados 0039", rotated_img)
cv2.waitKey(0)

# ==============================================================================
# 9. VOLTEO / ESPEJO DE IMÁGENES (FLIPPING)
# ==============================================================================
print("\n9. VOLTEANDO IMÁGENES")

horizontal_flip = cv2.flip(IMG_TOKEN, 1)

cv2.imshow("Volteo Horizontal 0039", horizontal_flip)
cv2.waitKey(0)

vertical_flip = cv2.flip(IMG_TOKEN, 0)

cv2.imshow("Volteo Vertical 0039", vertical_flip)
cv2.waitKey(0)

both_flip = cv2.flip(IMG_TOKEN, -1)

cv2.imshow("Volteo en Ambos Ejes 0039", both_flip)
cv2.waitKey(0)

# ==============================================================================
# 10. GUARDADO DE RESULTADOS Y LIMPIEZA
# ==============================================================================
print("\n10. GUARDANDO IMAGEN RESULTANTE")

ruta_guardado = os.path.join(ruta, "rotated_image_0039.jpg")

result = cv2.imwrite(ruta_guardado, rotated_img)

if result:
    print("Imagen rotada guardada exitosamente como rotated_image_0039.jpg")
else:
    print("Error: No se pudo guardar la imagen rotada.")

cv2.destroyAllWindows()

print("\n¡Todas las operaciones se completaron con éxito!")