# PANDUAN MENAMBAHKAN METRIK BARU

## 🎯 Pengantar
Panduan ini menjelaskan cara menambahkan metrik kualitas gambar baru ke dalam program analisis rontgen.

## 📋 Langkah-langkah Menambahkan Metrik Baru

### 1. Buat Fungsi Kalkulasi Metrik Baru
Tambahkan fungsi baru di bagian "FUNGSI KALKULASI METRIK" (sekitar baris 10-100):

```python
def calculate_your_metric(image, **kwargs):
    """
    Menghitung metrik kualitas gambar [NAMA METRIK].
    
    Args:
        image: Input gambar (numpy array), biasanya grayscale
        **kwargs: Parameter tambahan yang mungkin diperlukan
    
    Returns:
        float: Nilai metrik yang dihitung
    """
    try:
        # Pastikan gambar dalam format yang benar
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        
        # IMPLEMENTASI KALKULASI METRIK ANDA DI SINI
        # Contoh sederhana - menghitung rata-rata intensitas:
        metric_value = np.mean(gray_img)
        
        return float(metric_value)
        
    except Exception as e:
        print(f"⚠️  Error menghitung metrik: {e}")
        return np.nan
```

### 2. Tambahkan ke Fungsi calculate_all_metrics()
Di fungsi `calculate_all_metrics()` (sekitar baris 400), tambahkan perhitungan metrik baru:

```python
# Di dalam loop "# 1. Hitung ENT dan EME"
row_data[f"ENT {folder_name}"] = calculate_entropy(gray_img)
row_data[f"EME {folder_name}"] = calculate_eme(gray_img, r=eme_r, c=eme_c)
# TAMBAHKAN BARIS INI:
row_data[f"YOUR_METRIC {folder_name}"] = calculate_your_metric(gray_img)
```

### 3. Tambahkan ke Fungsi process_single_image()
Di fungsi `process_single_image()` (sekitar baris 500), tambahkan juga:

```python
# Di dalam loop "# 1. Hitung ENT dan EME"
row_data[f"ENT {folder_name}"] = calculate_entropy(gray_img)
row_data[f"EME {folder_name}"] = calculate_eme(gray_img, r=eme_r, c=eme_c)
# TAMBAHKAN BARIS INI:
row_data[f"YOUR_METRIC {folder_name}"] = calculate_your_metric(gray_img)
```

### 4. KHUSUS: Untuk Metrik Perbandingan Antar Folder
Jika metrik Anda membandingkan dua gambar dari folder berbeda (seperti PSNR, SSIM), tambahkan di bagian CII:

```python
# Di dalam loop "# 2. Hitung CII antar folder"
row_data[f"CII {proc_name} vs {ref_name}"] = calculate_cii(proc_gray, ref_gray, mask)
row_data[f"CII {ref_name} vs {proc_name}"] = calculate_cii(ref_gray, proc_gray, mask)

# TAMBAHKAN METRIK PERBANDINGAN ANDA DI SINI:
row_data[f"PSNR {proc_name} vs {ref_name}"] = calculate_psnr(proc_gray, ref_gray)
row_data[f"PSNR {ref_name} vs {proc_name}"] = calculate_psnr(ref_gray, proc_gray)
```

⚠️ **PENTING**: Jangan lupa tambahkan juga di fungsi `process_single_image()` untuk pemrosesan paralel!

### 5. Perbarui Pengorganisasian Kolom Output
Di fungsi `main()` (sekitar baris 670), tambahkan sorting untuk kolom baru:

```python
cols = ["no gambar"]
ent_cols = sorted([col for col in df.columns if col.startswith("ENT")])
eme_cols = sorted([col for col in df.columns if col.startswith("EME")])
cii_cols = sorted([col for col in df.columns if col.startswith("CII")])
# TAMBAHKAN BARIS INI:
your_metric_cols = sorted([col for col in df.columns if col.startswith("YOUR_METRIC")])

# Dan perbarui reindex:
df = df.reindex(columns=cols + ent_cols + eme_cols + your_metric_cols + cii_cols)
```

## 🔧 Contoh Implementasi Lengkap

### Contoh 1: Metrik Antar Folder - PSNR (Peak Signal-to-Noise Ratio)
**PSNR membandingkan dua gambar, jadi ditambahkan di bagian CII:**

```python
def calculate_psnr(image1, image2, max_val=255.0):
    """
    Menghitung Peak Signal-to-Noise Ratio antara dua gambar.
    Nilai tinggi = gambar lebih mirip (kualitas lebih baik)
    """
    try:
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')  # Gambar identik
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return float(psnr)
    except Exception as e:
        print(f"⚠️  Error menghitung PSNR: {e}")
        return np.nan

# Tambahkan di 2 tempat:
# 1. Di calculate_all_metrics() bagian "# 2. Hitung CII antar folder"
# 2. Di process_single_image() bagian yang sama
```

### Contoh 2: Menambahkan Metrik Variance
```python
def calculate_variance(image):
    """
    Menghitung varians intensitas piksel gambar.
    """
    try:
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        
        variance = np.var(gray_img.astype(float))
        return float(variance)
    except Exception as e:
        print(f"⚠️  Error menghitung Variance: {e}")
        return np.nan
```

## 📝 Template Cepat untuk Metrik Sederhana

```python
def calculate_NAMA_METRIK(image):
    """
    Menghitung [DESKRIPSI METRIK].
    """
    try:
        # Konversi ke grayscale jika perlu
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        
        # KALKULASI METRIK ANDA
        result = 0.0  # Ganti dengan kalkulasi sebenarnya
        
        return float(result)
    except Exception as e:
        print(f"⚠️  Error menghitung NAMA_METRIK: {e}")
        return np.nan
```

## 📊 Jenis-jenis Metrik dan Tempat Penambahannya

### 1. **Metrik Gambar Tunggal** (seperti ENT, EME, Variance)
- **Tambahkan di**: Bagian "# 1. Hitung ENT dan EME"
- **Contoh**: `row_data[f"VAR {folder_name}"] = calculate_variance(gray_img)`
- **Karakteristik**: Menganalisis satu gambar saja

### 2. **Metrik Perbandingan Antar Folder** (seperti CII, PSNR, SSIM)
- **Tambahkan di**: Bagian "# 2. Hitung CII antar folder"
- **Contoh**: `row_data[f"PSNR {proc_name} vs {ref_name}"] = calculate_psnr(proc_gray, ref_gray)`
- **Karakteristik**: Membandingkan gambar dari 2 folder berbeda

### 3. **Metrik dengan Parameter Khusus**
- **Tambahkan parameter** di `get_processing_options()`
- **Gunakan parameter** dari `options` dictionary
- **Contoh**: EME menggunakan parameter `r` dan `c`

## ⚠️  Hal-hal Penting yang Harus Diperhatikan

1. **Penamaan Konsisten**: Gunakan penamaan yang konsisten untuk metrik Anda
2. **Error Handling**: Selalu tambahkan try-catch untuk menangani error
3. **Format Output**: Pastikan return value dalam format float atau np.nan
4. **Dokumentasi**: Tambahkan docstring yang jelas untuk fungsi Anda
5. **Testing**: Test dengan berbagai jenis gambar sebelum digunakan
6. **Dual Implementation**: Untuk metrik antar folder, tambahkan di 2 fungsi (sequential & parallel)

## 🧪 Testing Metrik Baru

1. Buat folder test dengan beberapa gambar rontgen
2. Jalankan program dengan metrik baru
3. Periksa hasil CSV apakah kolom metrik baru muncul
4. Verifikasi nilai metrik masuk akal

## 📚 Referensi Metrik Umum

- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio  
- **MSE**: Mean Squared Error
- **Variance**: Varians intensitas
- **Standard Deviation**: Deviasi standar
- **Skewness**: Kemencengan distribusi
- **Kurtosis**: Keruncingan distribusi

## 💡 Tips Tambahan

1. Untuk metrik yang membutuhkan dua gambar (seperti PSNR, SSIM), implementasinya seperti CII
2. Untuk metrik yang membutuhkan parameter tambahan, tambahkan ke `get_processing_options()`
3. Gunakan library seperti `skimage.metrics` untuk implementasi metrik standar
4. Test performa untuk dataset besar sebelum implementasi final
