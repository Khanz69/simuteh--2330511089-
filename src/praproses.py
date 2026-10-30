
"""Pipeline praproses citra SIMUTEH."""

import json
import os
import cv2
import numpy as np

PARAM_BAWAAN = {
    "lebar": 640,
    "gamma": 1.0,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "penapis": "median",
    "kernel": 3,
    "sigma": 1.0,
    "tajam_k": 0.0,
}


def praproses(img_bgr, p=None):
    """Resize -> gamma -> CLAHE V -> filter -> unsharp masking."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input citra kosong atau tidak valid.")

    p = {**PARAM_BAWAAN, **(p or {})}
    img = img_bgr.copy()

    # 1. Resize dengan rasio aspek tetap
    lebar = int(p["lebar"])
    if lebar <= 0:
        raise ValueError("lebar harus lebih besar dari nol.")

    h, w = img.shape[:2]
    if w != lebar:
        tinggi = max(1, round(h * lebar / w))
        metode = cv2.INTER_AREA if lebar < w else cv2.INTER_LINEAR
        img = cv2.resize(img, (lebar, tinggi), interpolation=metode)

    # 2. Koreksi gamma
    gamma = float(p["gamma"])
    if gamma <= 0:
        raise ValueError("gamma harus lebih besar dari nol.")

    if gamma != 1.0:
        lut = np.array([
            round(255 * ((i / 255.0) ** gamma))
            for i in range(256)
        ], dtype=np.uint8)
        img = cv2.LUT(img, lut)

    # 3. CLAHE pada kanal V HSV
    clip = float(p["clahe_clip"])
    if clip > 0:
        tile = int(p["clahe_tile"])
        if tile <= 0:
            raise ValueError("clahe_tile harus lebih besar dari nol.")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(tile, tile)
        )
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. Reduksi derau
    jenis = str(p["penapis"]).lower()
    kernel = int(p["kernel"])

    if jenis != "none":
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("kernel harus bilangan ganjil positif.")

        if jenis == "rerata":
            img = cv2.blur(img, (kernel, kernel))
        elif jenis == "gaussian":
            sigma = float(p["sigma"])
            if sigma <= 0:
                raise ValueError("sigma harus lebih besar dari nol.")
            img = cv2.GaussianBlur(
                img, (kernel, kernel), sigmaX=sigma
            )
        elif jenis == "median":
            img = cv2.medianBlur(img, kernel)
        else:
            raise ValueError(
                "penapis harus none, rerata, gaussian, atau median."
            )

    # 5. Unsharp masking, jika diaktifkan
    tajam_k = float(p["tajam_k"])
    if tajam_k < 0:
        raise ValueError("tajam_k tidak boleh negatif.")

    if tajam_k > 0:
        blur = cv2.GaussianBlur(img, (5, 5), 1.0)
        img = cv2.addWeighted(
            img, 1.0 + tajam_k, blur, -tajam_k, 0
        )

    return img


def simpan_param(p, path="../hasil/param_praproses.json"):
    """Simpan parameter agar eksperimen bisa diulang."""
    lengkap = {**PARAM_BAWAAN, **(p or {})}
    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(lengkap, f, indent=2, ensure_ascii=False)
