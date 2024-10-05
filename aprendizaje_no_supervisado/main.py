import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar los datos desde la ruta absoluta
ruta_csv = r'C:\Users\angel\Desktop\Metodos_de_aprendizaje\datasets\ciudades_colombia.csv'
df = pd.read_csv(ruta_csv, sep=';', encoding='latin1')

# Mostrar el DataFrame para verificar que los datos se cargaron correctamente
print(df)

# Gráfico de dispersión inicial
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperatura (°C)'], df['Precipitación (mm)'], color='gray', s=100, alpha=0.6)
plt.xlabel('Temperatura (°C)')
plt.ylabel('Precipitación (mm)')
plt.title('Dispersión de Ciudades de Colombia')
plt.grid(True)
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=3).fit(df[['Temperatura (°C)', 'Precipitación (mm)']])
centroids = kmeans.cluster_centers_

# Mostrar los centroides
print("Centroides:", centroids)

# Gráfico de dispersión para las ciudades con colores por clúster
plt.figure(figsize=(10, 6))  # Aumentar el tamaño de la figura
colors = ['blue', 'green', 'orange']
for i in range(3):
    cluster_data = df[kmeans.labels_ == i]
    plt.scatter(cluster_data['Temperatura (°C)'], cluster_data['Precipitación (mm)'], 
                c=colors[i], label=f'Clúster {i+1}', s=50, alpha=0.7)

# Etiquetar las ciudades
for i, row in df.iterrows():
    plt.text(row['Temperatura (°C)'], row['Precipitación (mm)'], row['Ciudad'], fontsize=9, ha='right')

# Gráfico de los centroides con diferentes colores y etiquetas
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroides')

# Etiquetar los centroides
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], f'C{i+1}', fontsize=12, weight='bold', ha='center')

# Añadir etiquetas y título
plt.xlabel('Temperatura (°C)')
plt.ylabel('Precipitación (mm)')
plt.title('Clustering KMeans: Temperatura y Precipitación en Ciudades de Colombia')
plt.legend()

# Mostrar el gráfico
plt.show()
