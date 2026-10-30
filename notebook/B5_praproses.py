
import os
import sys
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

sys.modules.pop("utils_cv", None)
sys.modules.pop("praproses", None)

from utils_cv import baca, kontras_bercak
from praproses import praproses, simpan_param

label = pd.read_csv(os.path.join(DATA, "label.csv"))

# Parameter awal untuk diuji, bukan klaim bahwa ini sudah paling optimal.
param = {
    "lebar": 640,
    "gamma": 1.0,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "penapis": "median",
    "kernel": 3,
    "sigma": 1.0,
    "tajam_k": 0.0
}

# Pilih kanal berdasarkan hasil Bagian 2.
# 0 = B, 1 = G, 2 = R. Ganti bila hasil eksperimenmu menunjukkan kanal lain lebih baik.
KANAL = 2

simpan_param(
    param,
    os.path.join(HASIL, "param_praproses.json")
)

print("Parameter tersimpan:")
print(param)
