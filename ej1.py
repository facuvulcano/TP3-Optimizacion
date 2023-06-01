import glob
import cv2
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt

directorio_imagenes = "."

imagenes = glob.glob(os.path.join(directorio_imagenes, "*.jpeg"))

matrices_imagenes = []

# Recorrer cada imagen y cargarla como una matriz
for imagen_ruta in imagenes:
    imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE)
    
    # Convertir la imagen en una matriz y apilarla en la lista
    matriz_imagen = np.array(imagen).reshape(-1)
    matrices_imagenes.append(matriz_imagen)

# Apilar las matrices de imágenes en una matriz de datos
matriz_datos = np.stack(matrices_imagenes)

# descomposición en valores singulares (SVD) de la matriz de datos

U, S, V = np.linalg.svd(matriz_datos)

# Dimensiones de la imagen
p = int(np.sqrt(matriz_datos.shape[0]))

# primeros y últimos autovectores de U
primeros_autovectores = U[:, :p]
ultimos_autovectores = U[:, -p:]

# Define el tamaño de la figura
plt.figure(figsize=(10, 5))

# Visualización de los primeros autovectores
plt.subplot(1, 2, 1)
plt.title("Primeros Autovectores")
for i in range(p):
    plt.subplot(1, p, i+1)
    autovector = primeros_autovectores[:, i].reshape(p, p)
    plt.imshow(autovector, cmap='gray')
    plt.axis('off')

# Visualización de los últimos autovectores
plt.subplot(1, 2, 2)
plt.title("Últimos Autovectores")
for i in range(p):
    plt.subplot(1, p, i+1)
    autovector = ultimos_autovectores[:, i].reshape(p, p)
    plt.imshow(autovector, cmap='gray')
    plt.axis('off')

# Ajusta los espacios entre subplots
plt.tight_layout()

plt.show()

# Selección del número mínimo de dimensiones (d) para comprimir una imagen
imagen_comprimir = matrices_imagenes[0]  # Tomamos la primera imagen para ejemplificar
error_tolerable = 0.05  # 5% de error tolerable

# Cálculo del número mínimo de dimensiones (d)
d = np.sum(np.cumsum(S) / np.sum(S) < (1 - error_tolerable)) + 1

# Compresión de la imagen
imagen_comprimida = U[:, :d] @ np.diag(S[:d]) @ V[:d, :]
imagen_comprimida = imagen_comprimida.reshape(p, p)

# Cálculo del error de compresión
error = np.linalg.norm(matriz_datos[0] - imagen_comprimida.flatten()) / np.linalg.norm(matriz_datos[0])

# Imprimir el número mínimo de dimensiones y el error de compresión
print("Número mínimo de dimensiones (d):", d)
print("Error de compresión:", error)
