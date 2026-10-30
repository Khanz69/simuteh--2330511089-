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

from utils_cv import baca, tampil, kontras_bercak

nama = "teh_041.png"

img = baca(os.path.join(DATA, "citra", nama))
md = cv2.imread(os.path.join(DATA, "mask_daun", nama), 0)
mb = cv2.imread(os.path.join(DATA, "mask_bercak", nama), 0)

if md is None or mb is None:
    raise FileNotFoundError("Masker daun atau masker bercak tidak ditemukan.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

b, g, r = cv2.split(img)
h, s, v = cv2.split(hsv)

kanal = {
    "B": b,
    "G": g,
    "R": r,
    "Gray": gray,
    "H": h,
    "S": s,
    "V": v
}

tampil(
    [img] + list(kanal.values()),
    ["Asli"] + list(kanal.keys()),
    baris=2,
    ukuran=(16, 8)
)

plt.savefig(
    os.path.join(HASIL, "b2_panel_kanal_teh041.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.show()

for nm, c in kanal.items():
    print(f"KB kanal {nm:4s}: {kontras_bercak(c, md, mb):.2f}")from google.colab import drive
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

from utils_cv import baca, tampil, kontras_bercak

nama = "teh_041.png"

img = baca(os.path.join(DATA, "citra", nama))
md = cv2.imread(os.path.join(DATA, "mask_daun", nama), 0)
mb = cv2.imread(os.path.join(DATA, "mask_bercak", nama), 0)

if md is None or mb is None:
    raise FileNotFoundError("Masker daun atau masker bercak tidak ditemukan.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

b, g, r = cv2.split(img)
h, s, v = cv2.split(hsv)

kanal = {
    "B": b,
    "G": g,
    "R": r,
    "Gray": gray,
    "H": h,
    "S": s,
    "V": v
}

tampil(
    [img] + list(kanal.values()),
    ["Asli"] + list(kanal.keys()),
    baris=2,
    ukuran=(16, 8)
)

plt.savefig(
    os.path.join(HASIL, "b2_panel_kanal_teh041.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.show()

for nm, c in kanal.items():
    print(f"KB kanal {nm:4s}: {kontras_bercak(c, md, mb):.2f}")
