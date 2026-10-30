
from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = "/content/drive/MyDrive/simuteh-2330511089-"
SRC = os.path.join(PROJECT, "src")
DATA = os.path.join(PROJECT, "data", "sintetis")
HASIL = os.path.join(PROJECT, "hasil")

os.makedirs(HASIL, exist_ok=True)
sys.path.insert(0, SRC)

if "utils_cv" in sys.modules:
    del sys.modules["utils_cv"]

from utils_cv import baca, statistik, kontras_bercak

print("Persiapan B3 berhasil.")
print("Data:", DATA)
print("Hasil:", HASIL)
