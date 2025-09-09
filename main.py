import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import logging

# --- FUNGSI KALKULASI METRIK ---
# Fungsi-fungsi ini sekarang menjadi bagian dari script utama.


def calculate_contrast(image, mask):
    """
    Menghitung kontras dari sebuah area pada gambar yang ditentukan oleh mask.
    """
    foreground = image[mask == 1]
    background = image[mask == 0]

    if foreground.size == 0 or background.size == 0:
        return 0.0

    X_f = np.mean(foreground)
    X_b = np.mean(background)

    if X_f + X_b == 0:
        return 0.0  # Menghindari pembagian dengan nol

    contrast = (X_f - X_b) / (X_f + X_b)
    return contrast


def calculate_cii(processed_image, reference_image, mask):
    """
    Menghitung Contrast Improvement Index (CII).
    """
    C_processed = calculate_contrast(processed_image, mask)
    C_reference = calculate_contrast(reference_image, mask)

    if C_reference == 0:
        return np.nan  # Tidak terdefinisi jika kontras referensi adalah 0

    CII = C_processed / C_reference
    return CII


def calculate_entropy(image):
    """
    Menghitung entropi dari sebuah gambar dengan dukungan berbagai bit depth.
    """
    # Tentukan jumlah bins berdasarkan tipe data
    if image.dtype == np.uint8:
        bins = 256
        range_vals = [0, 256]
    elif image.dtype == np.uint16:
        bins = 65536
        range_vals = [0, 65536]
    else:
        # Untuk tipe lain, normalisasi ke 8-bit
        image = cv2.convertScaleAbs(image)
        bins = 256
        range_vals = [0, 256]

    hist = cv2.calcHist([image], [0], None, [bins], range_vals)

    # Normalisasi histogram
    hist = hist / hist.sum()

    # Menghitung entropi (hapus nilai 0 untuk menghindari log(0))
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    return entropy


def calculate_eme(image, r, c, epsilon=1e-4):
    """
    Menghitung Effective Measure of Enhancement (EME) dari sebuah gambar.
    """
    height, width = image.shape
    block_height = height // r
    block_width = width // c

    eme = 0.0
    num_blocks = 0
    for i in range(r):
        for j in range(c):
            block = image[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ]

            if block.size == 0:
                continue

            I_max = float(np.max(block))
            I_min = float(np.min(block))

            if I_min + epsilon == 0:
                continue

            CR = (I_max + epsilon) / (I_min + epsilon)

            if CR > 1:
                eme += 20 * np.log(CR)
                num_blocks += 1

    if num_blocks == 0:
        return 0.0

    eme /= num_blocks
    return eme


# --- LOGIKA PROGRAM UTAMA ---

# Konfigurasi logging untuk menekan pesan yang tidak perlu
logging.basicConfig(level=logging.WARNING)


def get_folder_paths():
    """Meminta input path folder dari user secara berulang."""
    folder_paths = []
    print("--- Masukkan Path Folder Gambar ---")
    print("Ketik 's' atau 'selesai' untuk berhenti memasukkan path.")

    while True:
        path = (
            input(f"Masukkan path folder ke-{len(folder_paths) + 1}: ")
            .strip()
            .strip('"')
        )  # Hapus kutip jika ada

        if path.lower() in ["s", "selesai"]:
            if not folder_paths:
                print("Tidak ada folder yang dimasukkan. Program berhenti.")
                return None
            break

        if os.path.isdir(path):
            folder_paths.append(path)
            print(f"-> Folder '{os.path.basename(path)}' ditambahkan.")
        else:
            print(f"Error: Path '{path}' tidak valid. Silakan coba lagi.")

    if len(folder_paths) > 5:
        total_images = sum(
            len(files)
            for _, _, files in itertools.chain.from_iterable(
                os.walk(p) for p in folder_paths
            )
        )
        print("\n[PERINGATAN]")
        print(
            f"Anda memasukkan {len(folder_paths)} folder dengan total sekitar {total_images} file."
        )
        print("Proses kalkulasi, terutama CII, mungkin akan memakan waktu cukup lama.")

    return folder_paths


def parse_image_files(folder_paths):
    """Memindai folder dan memetakan gambar berdasarkan nomornya."""
    image_map = {path: {} for path in folder_paths}
    all_image_numbers = set()

    print("\nMemindai file gambar...")
    for folder in tqdm(folder_paths, desc="Memindai Folder", unit="folder"):
        for filename in os.listdir(folder):
            try:
                parts = filename.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    img_num = int(parts[0])
                    img_path = os.path.join(folder, filename)
                    image_map[folder][img_num] = img_path
                    all_image_numbers.add(img_num)
            except (ValueError, IndexError):
                continue

    return image_map, sorted(list(all_image_numbers))


def calculate_all_metrics(folder_paths, image_map, all_image_numbers):
    """Menghitung semua metrik untuk gambar yang cocok dan menampilkan progress."""
    results = []
    folder_names = {path: os.path.basename(path) for path in folder_paths}
    cii_combinations = list(itertools.combinations(folder_paths, 2))

    for img_num in tqdm(all_image_numbers, desc="Menghitung Metrik", unit="gambar"):
        row_data = {"no gambar": img_num}
        images = {}

        for path in folder_paths:
            if img_num in image_map[path]:
                img_path = image_map[path][img_num]
                images[path] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # 1. Hitung ENT dan EME
        for path, img in images.items():
            folder_name = folder_names[path]
            row_data[f"ENT {folder_name}"] = calculate_entropy(img)
            row_data[f"EME {folder_name}"] = calculate_eme(img, r=4, c=4)

        # 2. Hitung CII antar folder
        for ref_path, proc_path in cii_combinations:
            ref_name = folder_names[ref_path]
            proc_name = folder_names[proc_path]

            if ref_path in images and proc_path in images:
                ref_img, proc_img = images[ref_path], images[proc_path]
                mask = np.ones_like(ref_img, dtype=np.uint8)

                row_data[f"CII {proc_name} vs {ref_name}"] = calculate_cii(
                    proc_img, ref_img, mask
                )
                row_data[f"CII {ref_name} vs {proc_name}"] = calculate_cii(
                    ref_img, proc_img, mask
                )

        results.append(row_data)

    return results


def main():
    """Fungsi utama untuk menjalankan seluruh proses."""
    try:
        folder_paths = get_folder_paths()
        if not folder_paths:
            return

        image_map, all_image_numbers = parse_image_files(folder_paths)

        if not all_image_numbers:
            print("\nTidak ada gambar dengan format 'nomor_namafile' yang ditemukan.")
            return

        results_data = calculate_all_metrics(folder_paths, image_map, all_image_numbers)

        output_filename = "metrics_results.csv"
        df = pd.DataFrame(results_data)

        cols = ["no gambar"]
        ent_cols = sorted([col for col in df.columns if col.startswith("ENT")])
        eme_cols = sorted([col for col in df.columns if col.startswith("EME")])
        cii_cols = sorted([col for col in df.columns if col.startswith("CII")])

        df = df.reindex(columns=cols + ent_cols + eme_cols + cii_cols)
        df.to_csv(output_filename, index=False, float_format="%.4f")

        print(
            f"\nProses selesai! Hasil disimpan di '{os.path.abspath(output_filename)}'"
        )

    except Exception as e:
        print(f"\nTerjadi error: {e}")
    finally:
        input("\nTekan Enter untuk keluar.")


if __name__ == "__main__":
    main()
