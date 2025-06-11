# Impor pustaka yang diperlukan ke dalam skrip
from minisom import MiniSom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Coba muat gambar dari file
try:
    img_original = Image.open('gambar.jpg')
except FileNotFoundError:
    print("Error: File 'gambar.jpg' tidak ditemukan. Pastikan file ada di folder yang sama dengan skrip.")
    exit()

# Ubah ukuran gambar agar proses lebih cepat
img_resized = img_original.resize((150, 150))

# Konversi gambar ke array NumPy dan normalisasi
img_array = np.array(img_resized) / 255.0

# Simpan bentuk asli gambar untuk digunakan nanti saat visualisasi
original_shape = img_array.shape

# Ubah bentuk array 3D menjadi daftar piksel 2D
# (tinggi * lebar, 3 channel warna)
pixels = img_array.reshape(-1, 3)

print("Data gambar berhasil disiapkan untuk SOM.")
print(f"Bentuk data piksel yang diratakan: {pixels.shape}")



# Tentukan jumlah warna akhir yang kita inginkan
n_colors = 16

# Inisialisasi model SOM
som = MiniSom(x=1, y=n_colors, # Peta 1x16 akan menghasilkan 16 warna
              input_len=3,      # 3 fitur input (R, G, B)
              sigma=0.1,        # Radius lingkungan neuron
              learning_rate=0.2,
              random_seed=42)   # Agar hasil bisa direproduksi

# Inisialisasi bobot dan latih model dengan data piksel
print("\nMemulai pelatihan SOM untuk menemukan warna dominan...")
som.train_random(data=pixels, num_iteration=100)
print("Pelatihan SOM selesai.")


# Gunakan SOM yang sudah dilatih untuk mengkuantisasi (segmentasi) gambar
print("Mengganti warna piksel asli dengan warna dari palet SOM...")
segmented_pixels = som.quantization(pixels)

# Ubah bentuk kembali data piksel menjadi gambar
segmented_image_array = segmented_pixels.reshape(original_shape)

# Tampilkan gambar asli dan hasil segmentasi berdampingan
print("Menampilkan hasil...")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Tampilkan gambar asli (yang sudah diubah ukurannya)
ax[0].imshow(img_array)
ax[0].set_title('Gambar Asli (Resized)')
ax[0].axis('off')

# Tampilkan gambar hasil segmentasi SOM
ax[1].imshow(segmented_image_array)
ax[1].set_title(f'Hasil Segmentasi SOM ({n_colors} Warna)')
ax[1].axis('off')

plt.tight_layout()
plt.show()