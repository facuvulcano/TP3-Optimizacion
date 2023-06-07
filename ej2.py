import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Paso 1: Cargar los datos del archivo dataset_clusters.csv
data = pd.read_csv('dataset_clusters.csv')
X = data.values

# Paso 2: Definir la lista de dimensiones a probar
dims = [2, 4, 20, X.shape[1]]

# Paso 3: Iterar sobre cada dimensión d
for d in dims:
    print(f"--- Dimensión d={d} ---")

    # Paso 4: Reducción de dimensionalidad usando SVD
    svd = TruncatedSVD(n_components=d)
    X_reduced = svd.fit_transform(X)

    # Paso 5: Aplicar K-means para encontrar los clusters
    kmeans = KMeans(n_clusters=3)  # Suponemos que hay 3 clusters
    kmeans.fit(X_reduced)
    labels = kmeans.labels_
    centroids_reduced = kmeans.cluster_centers_

    # Paso 6: Evaluar la calidad de los clusters
    silhouette_avg = silhouette_score(X_reduced, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Paso 7: Graficar los clusters y centroides en el espacio reducido
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels)
    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='x', color='red', s=100)
    plt.title(f'Clusters en el espacio reducido (d={d})')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.show()

    # Paso adicional: Transformar centroides al espacio original de alta dimensión
    centroids = svd.inverse_transform(centroids_reduced)

    # Paso adicional: Construir clasificador basado en la distancia a los centroides
    distances = cdist(X, centroids, metric='euclidean')
    cluster_assignment = np.argmin(distances, axis=1)

    # Paso 8: Asignar etiquetas a cada muestra en el dataset original
    data[f'Cluster_{d}'] = cluster_assignment

# Paso 9: Imprimir el dataframe con las etiquetas asignadas
print(data)