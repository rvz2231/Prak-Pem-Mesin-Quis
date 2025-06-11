# Impor pustaka yang diperlukan
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Gunakan data 'Mall_Customers.csv'
dataset = pd.read_csv('Mall_Customers.csv')

# 2. Tentukan fitur yang tepat untuk clustering
# Kita memilih 'Annual Income' dan 'Spending Score'
# Kolom indeks ke-3 dan ke-4
X = dataset.iloc[:, [3, 4]].values

print("Data berhasil dimuat dan fitur telah dipilih.")
print("Lima baris pertama dataset:")
print(dataset.head())


# List untuk menyimpan nilai WCSS untuk setiap k
wcss = []

# Perulangan untuk k dari 1 hingga 10
for i in range(1, 11):
    # Inisialisasi KMeans untuk setiap k
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init='auto')
    # Melatih model
    kmeans.fit(X)
    # Menyimpan nilai WCSS (diakses melalui atribut inertia_)
    wcss.append(kmeans.inertia_)

# Membuat plot Metode Elbow
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()



# 3. Buatlah model K-Means dengan mempertimbangkan jumlah k yang terbaik.
# Dari Metode Elbow, k terbaik adalah 5
kmeans_final = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init='auto')
y_kmeans = kmeans_final.fit_predict(X)

# Visualisasi hasil cluster
plt.figure(figsize=(12, 7))

# Plot data poin untuk setiap cluster dengan warna berbeda
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='cyan', label='Target Pelit')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='green', label='Rata-rata')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='magenta', label='Target Utama')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='blue', label='Boros')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='orange', label='Hemat')

# Plot centroid
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], s=200, c='red', label='Centroids', marker='X')

# Memberikan judul dan label
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()