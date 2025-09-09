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
#
# 📝 CARA MENAMBAHKAN METRIK BARU:
# 1. Tambahkan fungsi calculate_your_metric() di bagian ini
# 2. Ikuti template yang sama: input image, output float/np.nan
# 3. Tambahkan error handling dengan try-catch
# 4. Lihat file HOW_TO_ADD_METRICS.md untuk panduan lengkap


def calculate_contrast(image, mask):
    """
    Menghitung kontras dari sebuah area pada gambar yang ditentukan oleh mask.
    """
    try:
        # Validasi input
        if image is None or mask is None:
            return 0.0

        if image.shape != mask.shape:
            return 0.0

        # Pastikan mask adalah binary (0 atau 1)
        mask_binary = (mask > 0).astype(np.uint8)

        foreground = image[mask_binary == 1]
        background = image[mask_binary == 0]

        # Cek apakah ada piksel di foreground dan background
        if foreground.size == 0 or background.size == 0:
            return 0.0

        X_f = np.mean(foreground.astype(np.float64))
        X_b = np.mean(background.astype(np.float64))

        # Cek pembagian dengan nol
        denominator = X_f + X_b
        if abs(denominator) < 1e-10:
            return 0.0

        contrast = (X_f - X_b) / denominator

        # Pastikan hasil valid
        if np.isnan(contrast) or np.isinf(contrast):
            return 0.0

        return float(contrast)

    except Exception as e:
        print(f"⚠️  Error dalam kalkulasi kontras: {e}")
        return 0.0


def calculate_cii(processed_image, reference_image, mask):
    """
    Menghitung Contrast Improvement Index (CII) dengan validasi keamanan.
    """
    try:
        # Validasi input
        if processed_image is None or reference_image is None:
            return np.nan

        if mask is None:
            mask = np.ones_like(processed_image, dtype=np.uint8)

        # Pastikan mask memiliki ukuran yang sama
        if mask.shape != processed_image.shape:
            return np.nan

        C_processed = calculate_contrast(processed_image, mask)
        C_reference = calculate_contrast(reference_image, mask)

        # Validasi hasil kontras
        if np.isnan(C_processed) or np.isnan(C_reference):
            return np.nan

        # Validasi pembagi tidak nol
        if abs(C_reference) < 1e-10:  # Hampir nol
            return np.nan

        CII = C_processed / C_reference

        # Validasi hasil akhir
        if np.isnan(CII) or np.isinf(CII):
            return np.nan

        return float(CII)

    except Exception as e:
        print(f"⚠️  Error dalam kalkulasi CII: {e}")
        return np.nan


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


# 📝 TEMPLATE UNTUK MENAMBAHKAN METRIK BARU:
#
# def calculate_your_new_metric(image, **kwargs):
#     """
#     Menghitung metrik kualitas gambar [NAMA METRIK].
#
#     Args:
#         image: Input gambar (numpy array), biasanya grayscale
#         **kwargs: Parameter tambahan yang mungkin diperlukan
#
#     Returns:
#         float: Nilai metrik yang dihitung, atau np.nan jika error
#     """
#     try:
#         # Pastikan gambar dalam format grayscale
#         if len(image.shape) == 3:
#             gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_img = image
#
#         # IMPLEMENTASI KALKULASI METRIK ANDA DI SINI
#         # Contoh: menghitung rata-rata intensitas
#         result = np.mean(gray_img)
#
#         return float(result)
#
#     except Exception as e:
#         print(f"⚠️  Error menghitung metrik baru: {e}")
#         return np.nan


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
        print(f"⚠️  Peringatan: Gagal membuat area fokus analisis")
        print(f"   Program akan menganalisis seluruh gambar sebagai gantinya")
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
    Membaca file gambar dengan aman dan penanganan error yang baik.
    """
    try:
        if not os.path.exists(filepath):
            return None

        # Coba baca dengan OpenCV
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Coba metode alternatif untuk format khusus
            try:
                import imageio

                img = imageio.imread(filepath)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except:
                return None

        return img
    except Exception as e:
        print(f"⚠️  Peringatan: Gagal membaca file {os.path.basename(filepath)}")
        print(f"   Kemungkinan file rusak atau format tidak didukung")
        return None


# --- LOGIKA PROGRAM UTAMA ---

# Konfigurasi logging untuk menekan pesan yang tidak perlu
logging.basicConfig(level=logging.WARNING)


def get_folder_paths():
    """Meminta input path folder dari user secara berulang."""
    folder_paths = []
    print("=" * 50)
    print("📁 PILIH FOLDER GAMBAR RONTGEN")
    print("=" * 50)
    print("Masukkan alamat folder yang berisi gambar rontgen Anda.")
    print("Contoh: C:\\Users\\Nama\\Documents\\GambarRontgen")
    print("")
    print("⚠️  PENTING - FORMAT NAMA FILE:")
    print("   File gambar HARUS memiliki format: NOMOR_NAMAFILE")
    print("   Contoh yang BENAR:")
    print("   • 1_xray1.jpg")
    print("   • 2_rontgen_dada.png")
    print("   • 3_medical_image.tiff")
    print("")
    print("   Contoh yang SALAH:")
    print("   • xray1.jpg (tidak ada nomor)")
    print("   • 1-xray1.jpg (pakai tanda - bukan _)")
    print("   • gambar_1.jpg (nomor di belakang)")
    print("")
    print("💡 Tips: Anda bisa copy-paste alamat folder dari File Explorer")
    print("   Ketik 's' atau 'selesai' jika sudah selesai menambah folder")
    print("")

    while True:
        path = (
            input(f"📂 Alamat folder ke-{len(folder_paths) + 1}: ").strip().strip('"')
        )  # Hapus kutip jika ada

        if path.lower() in ["s", "selesai"]:
            if not folder_paths:
                print("❌ Belum ada folder yang dipilih. Program akan berhenti.")
                return None
            break

        if os.path.isdir(path):
            folder_paths.append(path)
            print(f"✅ Folder '{os.path.basename(path)}' berhasil ditambahkan.")
        else:
            print(f"❌ Alamat folder tidak ditemukan: '{path}'")
            print("   Silakan periksa kembali alamat folder Anda.")

    if len(folder_paths) > 3:
        total_images = sum(
            len(files)
            for _, _, files in itertools.chain.from_iterable(
                os.walk(p) for p in folder_paths
            )
        )
        print(f"\n⚠️  PERINGATAN:")
        print(
            f"   Anda memilih {len(folder_paths)} folder dengan sekitar {total_images} file."
        )
        print(f"   Proses analisis mungkin membutuhkan waktu yang cukup lama.")
        print(f"   Pastikan komputer Anda tidak dalam mode sleep/hibernate.")

    return folder_paths


def get_mask_method():
    """Meminta user memilih metode masking untuk CII."""
    print("\n" + "=" * 40)
    print("🎯 PILIHAN AREA ANALISIS GAMBAR")
    print("=" * 40)
    print("Program dapat menganalisis bagian tertentu dari gambar rontgen:")
    print("")
    print("1. 📊 Seluruh Gambar (Direkomendasikan untuk pemula)")
    print("   → Menganalisis semua bagian gambar rontgen")
    print("")
    print("2. 🔍 Otsu Otomatis")
    print("   → Program otomatis memilih area penting")
    print("")
    print("3. 🎨 Adaptif Cerdas")
    print("   → Menyesuaikan dengan kondisi gambar")
    print("")
    print("4. 📈 Berdasarkan Kecerahan (75%)")
    print("   → Fokus pada area terang gambar")
    print("")

    while True:
        try:
            choice = input(
                "🤔 Pilihan Anda (1-4) [tekan Enter untuk pilihan 1]: "
            ).strip()
            if not choice:
                print("✅ Dipilih: Seluruh Gambar")
                return "none"

            choice = int(choice)
            if choice == 1:
                print("✅ Dipilih: Seluruh Gambar")
                return "none"
            elif choice == 2:
                print("✅ Dipilih: Otsu Otomatis")
                return "otsu"
            elif choice == 3:
                print("✅ Dipilih: Adaptif Cerdas")
                return "adaptive"
            elif choice == 4:
                print("✅ Dipilih: Berdasarkan Kecerahan")
                return "percentile"
            else:
                print("❌ Nomor tidak valid. Silakan pilih angka 1, 2, 3, atau 4.")
        except ValueError:
            print("❌ Harap masukkan angka saja (1, 2, 3, atau 4).")


def get_processing_options():
    """Meminta user memilih opsi pemrosesan tambahan."""
    print("\n" + "=" * 45)
    print("⚙️  PENGATURAN ANALISIS GAMBAR")
    print("=" * 45)

    # EME parameters
    print("📐 Pengaturan Analisis Detail (EME)")
    print("   Nilai ini menentukan seberapa detail analisis dilakukan")
    print("   Nilai lebih tinggi = analisis lebih detail (tapi lebih lama)")
    print("")

    try:
        r = int(
            input("🔢 Pembagian vertikal gambar [biarkan kosong untuk nilai 4]: ")
            or "4"
        )
        c = int(
            input("🔢 Pembagian horizontal gambar [biarkan kosong untuk nilai 4]: ")
            or "4"
        )
        print(f"✅ Menggunakan pembagian: {r} x {c} blok")
    except ValueError:
        r, c = 4, 4
        print("✅ Menggunakan pengaturan standar: 4 x 4 blok")

    print("\n⚡ Kecepatan Pemrosesan")
    print("   Pemrosesan paralel dapat mempercepat analisis")
    print("   Cocok jika Anda memiliki banyak gambar (>50 gambar)")

    # Parallel processing
    use_parallel = (
        input("🚀 Aktifkan pemrosesan cepat? (y/n) [biarkan kosong untuk 'tidak']: ")
        .strip()
        .lower()
    )
    use_parallel = use_parallel in ["y", "yes", "ya"]

    if use_parallel:
        print("✅ Pemrosesan cepat diaktifkan")
    else:
        print("✅ Menggunakan pemrosesan standar")

    return {"eme_r": r, "eme_c": c, "use_parallel": use_parallel}


def get_supported_extensions():
    """Mengembalikan ekstensi file yang didukung."""
    return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm", ".dicom"}


def parse_image_files(folder_paths):
    """Memindai folder dan memetakan gambar berdasarkan nomornya."""
    image_map = {path: {} for path in folder_paths}
    all_image_numbers = set()
    supported_extensions = get_supported_extensions()

    print("\n🔍 Memindai file gambar...")
    invalid_files = []
    total_files = 0
    valid_files = 0

    for folder in tqdm(folder_paths, desc="Memindai Folder", unit="folder"):
        try:
            folder_name = os.path.basename(folder)
            for filename in os.listdir(folder):
                # Check file extension
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in supported_extensions:
                    continue

                total_files += 1

                try:
                    parts = filename.split("_", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        img_num = int(parts[0])
                        img_path = os.path.join(folder, filename)
                        # Verify file exists and is readable
                        if os.path.isfile(img_path):
                            image_map[folder][img_num] = img_path
                            all_image_numbers.add(img_num)
                            valid_files += 1
                    else:
                        # File doesn't follow naming convention
                        invalid_files.append(f"{folder_name}/{filename}")
                except (ValueError, IndexError):
                    invalid_files.append(f"{folder_name}/{filename}")
                    continue
        except OSError as e:
            print(f"⚠️  Tidak dapat mengakses folder: {os.path.basename(folder)}")
            print(f"   Kemungkinan folder terkunci atau tidak ada izin akses")
            continue

    # Report results
    print(f"\n📊 HASIL PEMINDAIAN:")
    print(f"   Total file gambar ditemukan: {total_files}")
    print(f"   File dengan format benar: {valid_files}")
    print(f"   File dengan format salah: {len(invalid_files)}")

    if invalid_files and len(invalid_files) <= 10:
        print(f"\n⚠️  File dengan format nama salah:")
        for invalid_file in invalid_files:
            print(f"   • {invalid_file}")
        print(f"   Ubah nama file tersebut ke format: NOMOR_NAMA")
    elif len(invalid_files) > 10:
        print(f"\n⚠️  Ada {len(invalid_files)} file dengan format nama salah")
        print(f"   Contoh: {invalid_files[0]}, {invalid_files[1]}, ...")
        print(f"   Ubah semua nama file ke format: NOMOR_NAMA")

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

                # 📝 TAMBAHKAN METRIK BARU DI SINI:
                # row_data[f"YOUR_METRIC {folder_name}"] = calculate_your_new_metric(gray_img)
            except Exception as e:
                print(
                    f"⚠️  Gagal menganalisis gambar {img_num} dari folder {folder_name}"
                )
                print(f"   Kemungkinan format gambar tidak sesuai atau file rusak")
                row_data[f"ENT {folder_name}"] = "ERROR"
                row_data[f"EME {folder_name}"] = "ERROR"

        # 2. Hitung CII antar folder
        # 📝 UNTUK METRIK YANG MEMBANDINGKAN ANTAR FOLDER (seperti CII, PSNR, SSIM):
        # Logika: folder yang diinput belakangan (processed) dibagi folder yang diinput lebih dulu (reference)
        for i, proc_path in enumerate(folder_paths):
            for j, ref_path in enumerate(folder_paths):
                if i <= j:  # Skip jika processed tidak lebih belakangan dari reference
                    continue

                proc_name = folder_names[proc_path]  # folder belakangan = processed
                ref_name = folder_names[ref_path]  # folder lebih dulu = reference

                if ref_path in images and proc_path in images:
                    ref_img, proc_img = images[ref_path], images[proc_path]

                    # Validate image compatibility
                    is_compatible, message = validate_image_compatibility(
                        ref_img, proc_img
                    )
                    if not is_compatible:
                        print(
                            f"⚠️  Gambar {img_num}: Tidak bisa dibandingkan antara {ref_name} dan {proc_name}"
                        )
                        print(f"   Alasan: Ukuran atau format gambar berbeda")
                        row_data[f"CII {proc_name}/{ref_name}"] = (
                            "TIDAK BISA DIBANDINGKAN"
                        )
                        continue

                try:
                    # Convert to grayscale if needed
                    if len(ref_img.shape) == 3:
                        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                        proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
                    else:
                        ref_gray = ref_img.copy()
                        proc_gray = proc_img.copy()

                    # Ensure images are in the same data type
                    if ref_gray.dtype != proc_gray.dtype:
                        # Convert to float64 for consistent calculation
                        ref_gray = ref_gray.astype(np.float64)
                        proc_gray = proc_gray.astype(np.float64)

                    # Create mask with error handling
                    try:
                        mask = create_roi_mask(ref_gray, method=mask_method)

                        # Validate mask
                        if mask is None or mask.shape != ref_gray.shape:
                            print(
                                f"⚠️  Gambar {img_num}: Mask tidak valid, menggunakan seluruh gambar"
                            )
                            mask = np.ones_like(ref_gray, dtype=np.uint8)
                    except Exception as mask_error:
                        print(
                            f"⚠️  Gambar {img_num}: Error membuat mask, menggunakan seluruh gambar"
                        )
                        mask = np.ones_like(ref_gray, dtype=np.uint8)

                    # Calculate CII: processed / reference
                    cii_result = calculate_cii(proc_gray, ref_gray, mask)

                    # Validate CII results
                    if np.isinf(cii_result):
                        print(
                            f"⚠️  Gambar {img_num}: CII {proc_name}/{ref_name} menghasilkan infinity"
                        )
                        cii_result = np.nan

                    row_data[f"CII {proc_name}/{ref_name}"] = cii_result

                    # 📝 TAMBAHKAN METRIK PERBANDINGAN ANTAR FOLDER DI SINI:
                    # Contoh untuk PSNR:
                    # row_data[f"PSNR {proc_name} vs {ref_name}"] = calculate_psnr(proc_gray, ref_gray)
                    # row_data[f"PSNR {ref_name} vs {proc_name}"] = calculate_psnr(ref_gray, proc_gray)
                except Exception as e:
                    print(
                        f"Warning: Failed to calculate CII for {proc_name}/{ref_name}, image {img_num}: {e}"
                    )
                    row_data[f"CII {proc_name}/{ref_name}"] = np.nan

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

            # 📝 TAMBAHKAN METRIK BARU DI SINI JUGA (untuk pemrosesan paralel):
            # row_data[f"YOUR_METRIC {folder_name}"] = calculate_your_new_metric(gray_img)
        except Exception as e:
            print(
                f"Warning: Failed to calculate metrics for {folder_name}, image {img_num}: {e}"
            )
            row_data[f"ENT {folder_name}"] = np.nan
            row_data[f"EME {folder_name}"] = np.nan

    # 2. Hitung CII antar folder
    # 📝 UNTUK METRIK PERBANDINGAN ANTAR FOLDER (seperti CII, PSNR, SSIM):
    # Tambahkan kalkulasi di bagian ini juga, untuk pemrosesan paralel
    for ref_path, proc_path in cii_combinations:
        ref_name = folder_names[ref_path]  # reference (earlier folder)
        proc_name = folder_names[proc_path]  # processed (later folder)

        if ref_path in images and proc_path in images:
            ref_img, proc_img = images[ref_path], images[proc_path]

            # Validate image compatibility
            is_compatible, message = validate_image_compatibility(ref_img, proc_img)
            if not is_compatible:
                print(f"Warning: Images incompatible for CII calculation ({message})")
                row_data[f"CII {proc_name}/{ref_name}"] = np.nan
                continue

            try:
                # Convert to grayscale if needed
                if len(ref_img.shape) == 3:
                    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
                else:
                    ref_gray = ref_img.copy()
                    proc_gray = proc_img.copy()

                # Ensure images are in the same data type
                if ref_gray.dtype != proc_gray.dtype:
                    # Convert to float64 for consistent calculation
                    ref_gray = ref_gray.astype(np.float64)
                    proc_gray = proc_gray.astype(np.float64)

                # Create mask with error handling
                try:
                    mask = create_roi_mask(ref_gray, method=mask_method)

                    # Validate mask
                    if mask is None or mask.shape != ref_gray.shape:
                        mask = np.ones_like(ref_gray, dtype=np.uint8)
                except Exception as mask_error:
                    mask = np.ones_like(ref_gray, dtype=np.uint8)

                # Calculate CII: processed / reference
                cii_result = calculate_cii(proc_gray, ref_gray, mask)

                # Validate CII results
                if np.isinf(cii_result):
                    cii_result = np.nan

                row_data[f"CII {proc_name}/{ref_name}"] = cii_result

                # 📝 TAMBAHKAN METRIK PERBANDINGAN ANTAR FOLDER DI SINI JUGA:
                # Contoh untuk PSNR:
                # row_data[f"PSNR {proc_name} vs {ref_name}"] = calculate_psnr(proc_gray, ref_gray)
                # row_data[f"PSNR {ref_name} vs {proc_name}"] = calculate_psnr(ref_gray, proc_gray)
            except Exception as e:
                print(
                    f"Warning: Failed to calculate CII for {proc_name}/{ref_name}, image {img_num}: {e}"
                )
                row_data[f"CII {proc_name}/{ref_name}"] = np.nan

    return row_data


def calculate_all_metrics_parallel(folder_paths, image_map, all_image_numbers, options):
    """
    Calculate metrics using parallel processing for better performance.
    """
    folder_names = {path: os.path.basename(path) for path in folder_paths}
    # Create combinations where later folder is processed, earlier is reference
    cii_combinations = []
    for i, proc_path in enumerate(folder_paths):
        for j, ref_path in enumerate(folder_paths):
            if i > j:  # Only if processed folder comes after reference folder
                cii_combinations.append((ref_path, proc_path))
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
        print(f"🚀 Menggunakan pemrosesan cepat dengan {max_workers} proses paralel...")
        print("   Silakan tunggu, proses sedang berjalan...")

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
        print("🏥" + "=" * 60 + "🏥")
        print("     PROGRAM ANALISIS KUALITAS GAMBAR RONTGEN")
        print("        Mengukur Entropi, EME, dan CII")
        print("🏥" + "=" * 60 + "🏥")
        print("")
        print("Selamat datang! Program ini akan membantu Anda menganalisis")
        print("kualitas gambar rontgen dengan menghitung metrik:")
        print("• ENT (Entropi)")
        print("• EME (Effective Measure of Enhancement)")
        print("• CII (Contrast Improvement Index)")
        print("")
        print("📋 PENTING - Format Nama File:")
        print("   Program akan membandingkan gambar berdasarkan nomor urut")
        print("   Contoh: 1_xray1.jpg di folder A vs 1_rontgen.jpg di folder B")
        print("   Format: NOMOR_NAMAFILE (nomor sama = gambar yang dibandingkan)")
        print("")
        print("🔧 Info untuk Developer:")
        print("   Untuk menambahkan metrik baru, lihat:")
        print("   • File HOW_TO_ADD_METRICS.md - panduan lengkap")
        print("   • Komentar 📝 di dalam kode - lokasi penambahan")
        print("   • File EXAMPLE_ADD_VARIANCE_METRIC.py - contoh implementasi")
        print("")

        folder_paths = get_folder_paths()
        if not folder_paths:
            return

        # Get user preferences
        mask_method = get_mask_method()
        processing_options = get_processing_options()
        processing_options["mask_method"] = mask_method

        image_map, all_image_numbers = parse_image_files(folder_paths)

        if not all_image_numbers:
            print("\n❌ TIDAK ADA GAMBAR DENGAN FORMAT YANG BENAR DITEMUKAN")
            print("")
            print("Silakan pastikan nama file gambar menggunakan format:")
            print("NOMOR_NAMAFILE (contoh: 1_xray1.jpg, 2_rontgen.png)")
            print("")
            print("Tips perbaikan:")
            print("• Ubah nama file menjadi: 1_[nama].jpg, 2_[nama].jpg, dst.")
            print("• Pastikan nomor dimulai dari 1, 2, 3, ... (tanpa nol di depan)")
            print("• Gunakan underscore (_) bukan tanda minus (-) atau spasi")
            print("")
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

        # 📝 TAMBAHKAN SORTING KOLOM METRIK BARU DI SINI:
        # your_metric_cols = sorted([col for col in df.columns if col.startswith("YOUR_METRIC")])

        # 📝 UNTUK MENGUBAH NAMA HEADER KOLOM:
        # df.columns = df.columns.str.replace('nama_panjang', 'nama_pendek')
        # Contoh: df.columns = df.columns.str.replace('_images', '')

        df = df.reindex(columns=cols + ent_cols + eme_cols + cii_cols)
        # 📝 JANGAN LUPA UPDATE REINDEX INI JUGA:
        # df = df.reindex(columns=cols + ent_cols + eme_cols + your_metric_cols + cii_cols)
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
