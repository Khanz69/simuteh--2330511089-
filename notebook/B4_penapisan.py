
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/drive/MyDrive/simuteh-2330511089-"
SRC = os.path.join(PROJECT, "src")
DATA = os.path.join(PROJECT, "data", "sintetis")
HASIL = os.path.join(PROJECT, "hasil")

os.makedirs(HASIL, exist_ok=True)
sys.path.insert(0, SRC)

# Muat ulang modul agar tidak memakai versi lama di memori Colab
sys.modules.pop("utils_cv", None)
from utils_cv import baca, kontras_bercak

print("Folder proyek:", PROJECT)
print("Folder dataset tersedia:", os.path.isdir(DATA))
print("Folder hasil:", HASIL)

def konvolusi(img, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(
        img.astype(float),
        ((ph, ph), (pw, pw)),
        mode="edge"
    )
    out = np.zeros(img.shape, dtype=float)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            jendela = pad[y:y+kh, x:x+kw]
            out[y, x] = np.sum(jendela * k)

    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def konvolusi_sejati(img, k):
    # Konvolusi sejati membalik kernel 180 derajat
    return konvolusi(img, np.flip(k))

nama = "teh_041.png"
bersih = cv2.cvtColor(
    baca(os.path.join(DATA, "bersih", nama)),
    cv2.COLOR_BGR2GRAY
)

kernel = {
    "rerata": np.ones((3, 3), dtype=float) / 9,
    "asimetris": np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=float)
}

for nm, k in kernel.items():
    t0 = time.perf_counter()
    manual = konvolusi(bersih, k)
    t_manual = time.perf_counter() - t0

    t0 = time.perf_counter()
    pustaka = cv2.filter2D(
        bersih, -1, k,
        borderType=cv2.BORDER_REPLICATE
    )
    t_cv = time.perf_counter() - t0

    sejati = konvolusi_sejati(bersih, k)

    beda = np.abs(
        manual.astype(int) - pustaka.astype(int)
    ).max()
    beda_sejati = np.abs(
        sejati.astype(int) - pustaka.astype(int)
    ).max()

    print(
        f"{nm}: manual vs filter2D={beda}; "
        f"konvolusi sejati vs filter2D={beda_sejati}; "
        f"waktu manual={t_manual:.3f} s; "
        f"waktu filter2D={t_cv:.5f} s"
    )

