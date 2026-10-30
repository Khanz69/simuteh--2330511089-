
# B1 - LINGKUNGAN DAN PEMERIKSAAN CITRA

from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt

# Lokasi project di Google Drive
PROJECT = "/content/drive/MyDrive/simuteh-2330511089-"
SRC = os.path.join(PROJECT, "src")
DATA = os.path.join(PROJECT, "data", "sintetis")
HASIL = os.path.join(PROJECT, "hasil")

# Pastikan folder src dapat ditemukan Python
sys.path.insert(0, SRC)

# Muat ulang utils_cv agar menggunakan file terbaru
if "utils_cv" in sys.modules:
    del sys.modules["utils_cv"]

from utils_cv import baca, tampil

# Informasi versi library
print(
    "OpenCV", cv2.__version__,
    "| NumPy", np.__version__,
    "| scikit-image", skimage.__version__
)

# Baca citra teh_041.png
img_path = os.path.join(DATA, "citra", "teh_041.png")
img = baca(img_path)

# Informasi citra
print("shape:", img.shape, "dtype:", img.dtype)
print("min-maks:", img.min(), img.max())
print("piksel (y=100, x=200) dalam urutan BGR:", img[100, 200])

# Tampilkan citra TANPA dan DENGAN konversi BGR ke RGB
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Tanpa konversi (salah)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Dengan konversi BGR ke RGB")
plt.axis("off")

plt.tight_layout()

# Simpan hasil ke folder hasil
os.makedirs(HASIL, exist_ok=True)
output_path = os.path.join(HASIL, "b1_bgr_rgb.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Hasil disimpan di:", output_path)

