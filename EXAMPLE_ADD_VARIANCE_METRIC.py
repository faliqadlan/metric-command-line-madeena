"""
CONTOH IMPLEMENTASI: Menambahkan Metrik Variance

File ini menunjukkan bagaimana menambahkan metrik Variance (varians intensitas piksel)
ke dalam program analisis gambar rontgen.

Ikuti langkah-langkah ini untuk menambahkan metrik baru:
"""

import cv2
import numpy as np


# LANGKAH 1: Buat fungsi kalkulasi metrik baru
def calculate_variance(image):
    """
    Menghitung varians intensitas piksel dari gambar.

    Varians mengukur seberapa tersebar nilai intensitas piksel.
    Nilai tinggi = variasi intensitas tinggi
    Nilai rendah = intensitas relatif seragam

    Args:
        image: Input gambar (numpy array)

    Returns:
        float: Nilai varians, atau np.nan jika error
    """
    try:
        # Pastikan gambar dalam format grayscale
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image

        # Hitung varians intensitas piksel
        variance = np.var(gray_img.astype(float))

        return float(variance)

    except Exception as e:
        print(f"⚠️  Error menghitung Variance: {e}")
        return np.nan


# LANGKAH 2: Tambahkan ke fungsi calculate_all_metrics()
# Di dalam loop "# 1. Hitung ENT dan EME", tambahkan:
# row_data[f"VAR {folder_name}"] = calculate_variance(gray_img)

# LANGKAH 3: Tambahkan ke fungsi process_single_image()
# Di bagian yang sama, tambahkan:
# row_data[f"VAR {folder_name}"] = calculate_variance(gray_img)

# LANGKAH 4: Perbarui pengorganisasian kolom di main()
# Tambahkan:
# var_cols = sorted([col for col in df.columns if col.startswith("VAR")])
# df = df.reindex(columns=cols + ent_cols + eme_cols + var_cols + cii_cols)

"""
CONTOH PENGGUNAAN LEBIH KOMPLEKS: Metrik dengan Parameter

Berikut contoh metrik yang membutuhkan parameter tambahan:
"""


def calculate_local_variance(image, window_size=5):
    """
    Menghitung rata-rata varians lokal menggunakan sliding window.

    Args:
        image: Input gambar
        window_size: Ukuran window untuk perhitungan lokal

    Returns:
        float: Rata-rata varians lokal
    """
    try:
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image

        h, w = gray_img.shape
        local_variances = []

        # Sliding window
        for i in range(0, h - window_size + 1, window_size):
            for j in range(0, w - window_size + 1, window_size):
                window = gray_img[i : i + window_size, j : j + window_size]
                local_var = np.var(window.astype(float))
                local_variances.append(local_var)

        # Rata-rata varians lokal
        avg_local_variance = np.mean(local_variances)

        return float(avg_local_variance)

    except Exception as e:
        print(f"⚠️  Error menghitung Local Variance: {e}")
        return np.nan


"""
TIPS IMPLEMENTASI:

1. TESTING: Selalu test dengan beberapa gambar berbeda
2. PENAMAAN: Gunakan nama singkat tapi jelas (VAR, LVAR, dll)
3. DOKUMENTASI: Jelaskan apa yang diukur metrik tersebut
4. ERROR HANDLING: Selalu tangani kemungkinan error
5. TIPE DATA: Pastikan return value dalam format float

CONTOH HASIL CSV:
no gambar,ENT folder1,EME folder1,VAR folder1,CII folder2 vs folder1
1,7.5432,12.3456,1234.56,1.0823
2,7.2341,11.9876,1456.78,1.1245
...
"""
