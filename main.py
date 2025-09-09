import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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


def create_roi_mask(image, method="otsu"):
    """
    Membuat mask ROI (Region of Interest) untuk analisis yang lebih akurat.

    Args:
        image: Input image
        method: Method untuk thresholding ("otsu", "adaptive", "percentile", "none")

    Returns:
        Binary mask array
    """
    if method == "none":
        return np.ones_like(image, dtype=np.uint8)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize to 8-bit for thresholding operations
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    try:
        if method == "otsu":
            _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            mask = cv2.adaptiveThreshold(
                gray,
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
        elif method == "percentile":
            threshold = np.percentile(gray, 75)
            mask = (gray > threshold).astype(np.uint8)
        else:
            mask = np.ones_like(gray, dtype=np.uint8)
    except Exception as e:
        print(f"Warning: Mask generation failed ({e}). Using full image mask.")
        mask = np.ones_like(gray, dtype=np.uint8)

    return mask


def validate_image_compatibility(img1, img2):
    """
    Memvalidasi kompatibilitas dua gambar untuk perhitungan metrik.
    """
    if img1 is None or img2 is None:
        return False, "One or both images are None"

    if img1.shape != img2.shape:
        return False, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    return True, "Compatible"


def safe_imread(filepath):
    """
    Safely read image with error handling and format support.
    """
    try:
        if not os.path.exists(filepath):
            return None

        # Try reading with OpenCV
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Try alternative approach for special formats
            try:
                import imageio

                img = imageio.imread(filepath)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except:
                return None

        return img
    except Exception as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        return None


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


def get_mask_method():
    """Meminta user memilih metode masking untuk CII."""
    print("\n--- Opsi Masking untuk CII ---")
    print("1. None (seluruh gambar)")
    print("2. Otsu thresholding")
    print("3. Adaptive thresholding")
    print("4. Percentile-based (75%)")

    while True:
        try:
            choice = input("Pilih metode masking (1-4) [default: 1]: ").strip()
            if not choice:
                return "none"

            choice = int(choice)
            if choice == 1:
                return "none"
            elif choice == 2:
                return "otsu"
            elif choice == 3:
                return "adaptive"
            elif choice == 4:
                return "percentile"
            else:
                print("Pilihan tidak valid. Masukkan angka 1-4.")
        except ValueError:
            print("Input tidak valid. Masukkan angka 1-4.")


def get_processing_options():
    """Meminta user memilih opsi pemrosesan tambahan."""
    print("\n--- Opsi Pemrosesan ---")

    # EME parameters
    try:
        r = int(input("Masukkan nilai r untuk EME [default: 4]: ") or "4")
        c = int(input("Masukkan nilai c untuk EME [default: 4]: ") or "4")
    except ValueError:
        r, c = 4, 4
        print("Menggunakan nilai default r=4, c=4")

    # Parallel processing
    use_parallel = (
        input("Gunakan pemrosesan paralel? (y/n) [default: n]: ").strip().lower()
    )
    use_parallel = use_parallel in ["y", "yes"]

    return {"eme_r": r, "eme_c": c, "use_parallel": use_parallel}


def get_supported_extensions():
    """Mengembalikan ekstensi file yang didukung."""
    return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm", ".dicom"}


def parse_image_files(folder_paths):
    """Memindai folder dan memetakan gambar berdasarkan nomornya."""
    image_map = {path: {} for path in folder_paths}
    all_image_numbers = set()
    supported_extensions = get_supported_extensions()

    print("\nMemindai file gambar...")
    for folder in tqdm(folder_paths, desc="Memindai Folder", unit="folder"):
        try:
            for filename in os.listdir(folder):
                # Check file extension
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in supported_extensions:
                    continue

                try:
                    parts = filename.split("_", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        img_num = int(parts[0])
                        img_path = os.path.join(folder, filename)
                        # Verify file exists and is readable
                        if os.path.isfile(img_path):
                            image_map[folder][img_num] = img_path
                            all_image_numbers.add(img_num)
                except (ValueError, IndexError):
                    continue
        except OSError as e:
            print(f"Warning: Cannot access folder {folder}: {e}")
            continue

    return image_map, sorted(list(all_image_numbers))


def calculate_all_metrics(folder_paths, image_map, all_image_numbers, options):
    """Menghitung semua metrik untuk gambar yang cocok dan menampilkan progress."""
    results = []
    folder_names = {path: os.path.basename(path) for path in folder_paths}
    cii_combinations = list(itertools.combinations(folder_paths, 2))

    eme_r = options.get("eme_r", 4)
    eme_c = options.get("eme_c", 4)
    mask_method = options.get("mask_method", "none")

    for img_num in tqdm(all_image_numbers, desc="Menghitung Metrik", unit="gambar"):
        row_data = {"no gambar": img_num}
        images = {}

        # Load images with error handling
        for path in folder_paths:
            if img_num in image_map[path]:
                img_path = image_map[path][img_num]
                img = safe_imread(img_path)
                if img is not None:
                    images[path] = img

        # Skip if no images loaded for this number
        if not images:
            continue

        # 1. Hitung ENT dan EME
        for path, img in images.items():
            folder_name = folder_names[path]
            try:
                # Convert to grayscale for metric calculations
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = img

                row_data[f"ENT {folder_name}"] = calculate_entropy(gray_img)
                row_data[f"EME {folder_name}"] = calculate_eme(
                    gray_img, r=eme_r, c=eme_c
                )
            except Exception as e:
                print(
                    f"Warning: Failed to calculate metrics for {folder_name}, image {img_num}: {e}"
                )
                row_data[f"ENT {folder_name}"] = np.nan
                row_data[f"EME {folder_name}"] = np.nan

        # 2. Hitung CII antar folder
        for ref_path, proc_path in cii_combinations:
            ref_name = folder_names[ref_path]
            proc_name = folder_names[proc_path]

            if ref_path in images and proc_path in images:
                ref_img, proc_img = images[ref_path], images[proc_path]

                # Validate image compatibility
                is_compatible, message = validate_image_compatibility(ref_img, proc_img)
                if not is_compatible:
                    print(
                        f"Warning: Images incompatible for CII calculation ({message})"
                    )
                    row_data[f"CII {proc_name} vs {ref_name}"] = np.nan
                    row_data[f"CII {ref_name} vs {proc_name}"] = np.nan
                    continue

                try:
                    # Convert to grayscale if needed
                    if len(ref_img.shape) == 3:
                        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                        proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
                    else:
                        ref_gray = ref_img
                        proc_gray = proc_img

                    # Create mask
                    mask = create_roi_mask(ref_gray, method=mask_method)

                    row_data[f"CII {proc_name} vs {ref_name}"] = calculate_cii(
                        proc_gray, ref_gray, mask
                    )
                    row_data[f"CII {ref_name} vs {proc_name}"] = calculate_cii(
                        ref_gray, proc_gray, mask
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to calculate CII for {ref_name} vs {proc_name}, image {img_num}: {e}"
                    )
                    row_data[f"CII {proc_name} vs {ref_name}"] = np.nan
                    row_data[f"CII {ref_name} vs {proc_name}"] = np.nan

        results.append(row_data)

    return results


def process_single_image(args):
    """
    Process metrics for a single image number - used for parallel processing.
    """
    img_num, folder_paths, image_map, folder_names, cii_combinations, options = args

    eme_r = options.get("eme_r", 4)
    eme_c = options.get("eme_c", 4)
    mask_method = options.get("mask_method", "none")

    row_data = {"no gambar": img_num}
    images = {}

    # Load images with error handling
    for path in folder_paths:
        if img_num in image_map[path]:
            img_path = image_map[path][img_num]
            img = safe_imread(img_path)
            if img is not None:
                images[path] = img

    # Skip if no images loaded for this number
    if not images:
        return None

    # 1. Hitung ENT dan EME
    for path, img in images.items():
        folder_name = folder_names[path]
        try:
            # Convert to grayscale for metric calculations
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img

            row_data[f"ENT {folder_name}"] = calculate_entropy(gray_img)
            row_data[f"EME {folder_name}"] = calculate_eme(gray_img, r=eme_r, c=eme_c)
        except Exception as e:
            print(
                f"Warning: Failed to calculate metrics for {folder_name}, image {img_num}: {e}"
            )
            row_data[f"ENT {folder_name}"] = np.nan
            row_data[f"EME {folder_name}"] = np.nan

    # 2. Hitung CII antar folder
    for ref_path, proc_path in cii_combinations:
        ref_name = folder_names[ref_path]
        proc_name = folder_names[proc_path]

        if ref_path in images and proc_path in images:
            ref_img, proc_img = images[ref_path], images[proc_path]

            # Validate image compatibility
            is_compatible, message = validate_image_compatibility(ref_img, proc_img)
            if not is_compatible:
                print(f"Warning: Images incompatible for CII calculation ({message})")
                row_data[f"CII {proc_name} vs {ref_name}"] = np.nan
                row_data[f"CII {ref_name} vs {proc_name}"] = np.nan
                continue

            try:
                # Convert to grayscale if needed
                if len(ref_img.shape) == 3:
                    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
                else:
                    ref_gray = ref_img
                    proc_gray = proc_img

                # Create mask
                mask = create_roi_mask(ref_gray, method=mask_method)

                row_data[f"CII {proc_name} vs {ref_name}"] = calculate_cii(
                    proc_gray, ref_gray, mask
                )
                row_data[f"CII {ref_name} vs {proc_name}"] = calculate_cii(
                    ref_gray, proc_gray, mask
                )
            except Exception as e:
                print(
                    f"Warning: Failed to calculate CII for {ref_name} vs {proc_name}, image {img_num}: {e}"
                )
                row_data[f"CII {proc_name} vs {ref_name}"] = np.nan
                row_data[f"CII {ref_name} vs {proc_name}"] = np.nan

    return row_data


def calculate_all_metrics_parallel(folder_paths, image_map, all_image_numbers, options):
    """
    Calculate metrics using parallel processing for better performance.
    """
    folder_names = {path: os.path.basename(path) for path in folder_paths}
    cii_combinations = list(itertools.combinations(folder_paths, 2))
    use_parallel = options.get("use_parallel", False)

    # Prepare arguments for each image
    args_list = []
    for img_num in all_image_numbers:
        args_list.append(
            (img_num, folder_paths, image_map, folder_names, cii_combinations, options)
        )

    results = []

    if use_parallel and len(args_list) > 1:
        # Use parallel processing
        max_workers = min(
            4, os.cpu_count() or 1
        )  # Limit workers to prevent memory issues
        print(f"Using parallel processing with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_image, args) for args in args_list
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing Images"
            ):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing image: {e}")
    else:
        # Sequential processing
        for args in tqdm(args_list, desc="Menghitung Metrik", unit="gambar"):
            result = process_single_image(args)
            if result is not None:
                results.append(result)

    return results


def main():
    """Fungsi utama untuk menjalankan seluruh proses."""
    try:
        folder_paths = get_folder_paths()
        if not folder_paths:
            return

        # Get user preferences
        mask_method = get_mask_method()
        processing_options = get_processing_options()
        processing_options["mask_method"] = mask_method

        image_map, all_image_numbers = parse_image_files(folder_paths)

        if not all_image_numbers:
            print("\nTidak ada gambar dengan format 'nomor_namafile' yang ditemukan.")
            return

        print(f"\nDitemukan {len(all_image_numbers)} nomor gambar unik")
        total_possible = len(all_image_numbers) * len(folder_paths)
        print(f"Memulai kalkulasi metrik untuk maksimal {total_possible} gambar...")

        if processing_options.get("use_parallel", False):
            results_data = calculate_all_metrics_parallel(
                folder_paths, image_map, all_image_numbers, processing_options
            )
        else:
            results_data = calculate_all_metrics(
                folder_paths, image_map, all_image_numbers, processing_options
            )

        if not results_data:
            print(
                "\nTidak ada data hasil kalkulasi yang berhasil. Periksa format file dan path folder."
            )
            return

        # Generate timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"metrics_results_{timestamp}.csv"

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
        print(f"Total baris data: {len(df)}")
        print(f"Kolom yang dihasilkan: {len(df.columns)}")

    except KeyboardInterrupt:
        print("\n\nProses dibatalkan oleh user.")
    except Exception as e:
        print(f"\nTerjadi error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        input("\nTekan Enter untuk keluar.")


if __name__ == "__main__":
    main()
