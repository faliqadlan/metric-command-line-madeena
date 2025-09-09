# 🏥 Program Analisis Kualitas Gambar Rontgen

Program berbasis Python untuk menganalisis kualitas gambar rontgen dengan menghitung berbagai metrik kualitas gambar seperti Entropi (ENT), Effective Measure of Enhancement (EME), dan Contrast Improvement Index (CII).

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Metrik yang Diukur](#-metrik-yang-diukur)
- [Persyaratan Sistem](#-persyaratan-sistem)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Format Nama File](#-format-nama-file)
- [Contoh Output](#-contoh-output)
- [Pengaturan Lanjutan](#-pengaturan-lanjutan)
- [Untuk Developer](#-untuk-developer)
- [Troubleshooting](#-troubleshooting)
- [Kontribusi](#-kontribusi)

## 🌟 Fitur Utama

- **Multi-Metrik**: Mengukur ENT, EME, dan CII secara bersamaan
- **Multi-Folder**: Dapat menganalisis beberapa folder sekaligus
- **Format Fleksibel**: Mendukung JPG, PNG, TIFF, BMP, DICOM
- **ROI Analysis**: Pilihan area analisis (seluruh gambar, Otsu, adaptif, percentile)
- **Parallel Processing**: Pemrosesan cepat untuk dataset besar
- **Error Handling**: Penanganan error yang robust
- **Progress Tracking**: Progress bar untuk monitoring proses
- **Timestamped Output**: File hasil dengan timestamp otomatis

## 📊 Metrik yang Diukur

### 1. **ENT (Entropi)**
- Mengukur kompleksitas dan detail informasi dalam gambar
- Nilai tinggi = gambar memiliki banyak detail
- Nilai rendah = gambar cenderung seragam

### 2. **EME (Effective Measure of Enhancement)**
- Mengukur tingkat perbaikan/enhancement gambar
- Nilai tinggi = enhancement lebih baik
- Menggunakan pembagian blok untuk analisis lokal

### 3. **CII (Contrast Improvement Index)**
- Membandingkan kontras antara dua gambar
- Format: `CII folder_processed/folder_reference`
- Nilai > 1 = kontras meningkat
- Nilai < 1 = kontras menurun

## 💻 Persyaratan Sistem

### Minimum:
- **OS**: Windows 10/11, macOS 10.14+, atau Linux Ubuntu 18.04+
- **Python**: 3.7 atau lebih baru
- **RAM**: 4GB (8GB direkomendasikan untuk dataset besar)
- **Storage**: 100MB untuk program + ruang untuk hasil analisis

### Dependencies:
```
opencv-python >= 4.5.0
numpy >= 1.19.0
pandas >= 1.2.0
tqdm >= 4.60.0
imageio >= 2.9.0 (opsional, untuk format khusus)
```

## 🔧 Instalasi

### Opsi 1: Instalasi Manual
```bash
# Clone atau download repository
git clone [repository-url]
cd metric-command-line-madeena

# Install dependencies
pip install opencv-python numpy pandas tqdm imageio
```

### Opsi 2: Menggunakan Requirements File
```bash
# Jika ada file requirements.txt
pip install -r requirements.txt
```

### Opsi 3: Virtual Environment (Direkomendasikan)
```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install opencv-python numpy pandas tqdm imageio
```

## 🚀 Cara Penggunaan

### 1. Jalankan Program
```bash
python main.py
```

### 2. Pilih Folder Gambar
- Program akan meminta alamat folder yang berisi gambar rontgen
- Anda dapat menambahkan beberapa folder
- Contoh: `C:\Users\Nama\Documents\GambarRontgen`
- Ketik `s` atau `selesai` untuk melanjutkan

### 3. Pilih Area Analisis
1. **Seluruh Gambar** (Direkomendasikan untuk pemula)
2. **Otsu Otomatis** - Program otomatis memilih area penting
3. **Adaptif Cerdas** - Menyesuaikan dengan kondisi gambar
4. **Berdasarkan Kecerahan (75%)** - Fokus pada area terang

### 4. Pengaturan Analisis
- **Pembagian EME**: Mengatur detail analisis (default: 4x4)
- **Pemrosesan Paralel**: Untuk dataset besar (>50 gambar)

### 5. Hasil
Program akan menghasilkan file CSV dengan timestamp:
`metrics_results_YYYYMMDD_HHMMSS.csv`

## 📁 Format Nama File

### ⚠️ PENTING: Format Wajib
File gambar **HARUS** menggunakan format: `NOMOR_NAMAFILE`

### ✅ Contoh yang Benar:
```
1_xray1.jpg
2_rontgen_dada.png
3_medical_image.tiff
10_scan_kepala.jpg
25_dental_xray.bmp
```

### ❌ Contoh yang Salah:
```
xray1.jpg           (tidak ada nomor)
1-xray1.jpg         (pakai tanda - bukan _)
gambar_1.jpg        (nomor di belakang)
01_xray.jpg         (ada nol di depan)
1 xray.jpg          (pakai spasi)
```

### 🔍 Mengapa Format Ini Penting?
- Program membandingkan gambar berdasarkan nomor urut
- Contoh: `1_original.jpg` dibandingkan dengan `1_enhanced.jpg`
- Nomor yang sama = gambar yang akan dibandingkan

## 📄 Contoh Output

### Struktur CSV:
```csv
no gambar,ENT original,ENT enhanced,EME original,EME enhanced,CII enhanced/original
1,7.5432,7.8234,12.3456,15.6789,1.2345
2,7.2341,7.6789,11.9876,14.5432,1.1234
3,7.8765,8.1234,13.4567,16.7890,1.3456
```

### Interpretasi Hasil:
- **ENT > 7**: Gambar memiliki detail yang baik
- **EME > 10**: Enhancement cukup efektif
- **CII > 1**: Kontras meningkat setelah enhancement
- **CII < 1**: Kontras menurun setelah enhancement

## ⚙️ Pengaturan Lanjutan

### Pemrosesan Paralel
- Aktifkan untuk dataset > 50 gambar
- Menggunakan maksimal 4 CPU cores
- Mempercepat proses hingga 3-4x lipat

### EME Block Division
- Default: 4x4 blok
- Nilai lebih tinggi = analisis lebih detail
- Nilai lebih rendah = pemrosesan lebih cepat

### ROI Methods
- **None**: Analisis seluruh gambar
- **Otsu**: Threshold otomatis berdasarkan histogram
- **Adaptive**: Threshold adaptif lokal
- **Percentile**: Fokus pada 75% area terbright

## 👨‍💻 Untuk Developer

### Menambahkan Metrik Baru
Lihat dokumentasi lengkap di:
- `HOW_TO_ADD_METRICS.md` - Panduan komprehensif
- `EXAMPLE_ADD_VARIANCE_METRIC.py` - Contoh implementasi
- `NAMING_FORMAT_GUIDE.md` - Panduan format nama file

### Template Metrik Baru:
```python
def calculate_your_metric(image, **kwargs):
    """
    Menghitung metrik kualitas gambar baru.
    """
    try:
        # Pastikan format grayscale
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        
        # Implementasi kalkulasi
        result = np.mean(gray_img)  # Contoh
        
        return float(result)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return np.nan
```

### Lokasi Penambahan:
1. **Single Image Metrics**: Bagian `# 1. Hitung ENT dan EME`
2. **Inter-Folder Metrics**: Bagian `# 2. Hitung CII antar folder`
3. **Column Organization**: Bagian sorting di `main()` function

## 🛠️ Troubleshooting

### Problem: "TIDAK ADA GAMBAR DENGAN FORMAT YANG BENAR"
**Solusi:**
- Periksa format nama file: `NOMOR_NAMAFILE`
- Pastikan ada underscore (_) setelah nomor
- Nomor harus di awal nama file

### Problem: "Tidak bisa dibandingkan antara folder"
**Solusi:**
- Pastikan ukuran gambar sama di semua folder
- Periksa format file (JPG, PNG, dll.)
- Pastikan file tidak corrupt

### Problem: Program berjalan lambat
**Solusi:**
- Aktifkan pemrosesan paralel
- Kurangi pembagian EME (misal: 2x2)
- Periksa ukuran file gambar

### Problem: Error saat membaca file
**Solusi:**
- Pastikan file tidak sedang dibuka aplikasi lain
- Periksa permission folder (read access)
- Coba format file yang berbeda

### Problem: Hasil CII selalu NaN
**Solusi:**
- Periksa kontras gambar (jangan terlalu gelap/terang)
- Coba metode ROI yang berbeda
- Pastikan mask tidak kosong

## 📊 Format File yang Didukung

### ✅ Didukung:
- **JPEG/JPG** - Format umum fotografi
- **PNG** - Format dengan transparansi
- **TIFF/TIF** - Format medis berkualitas tinggi
- **BMP** - Format Windows bitmap
- **DICOM/DCM** - Format medis standar

### ❌ Tidak Didukung:
- GIF, WEBP, SVG, PDF
- File video (MP4, AVI, dll.)
- File teks atau dokumen

## 🤝 Kontribusi

### Cara Berkontribusi:
1. Fork repository ini
2. Buat branch baru untuk fitur/perbaikan
3. Commit perubahan dengan pesan yang jelas
4. Push ke branch dan buat Pull Request

### Yang Bisa Dikontribusikan:
- Metrik kualitas gambar baru
- Perbaikan algoritma existing
- Optimisasi performa
- Dokumentasi dan tutorial
- Bug fixes dan error handling

### Coding Standards:
- Gunakan docstring untuk fungsi
- Tambahkan error handling
- Test dengan berbagai jenis gambar
- Follow existing code style

## 📞 Support & Kontak

### Jika Mengalami Masalah:
1. Periksa bagian [Troubleshooting](#-troubleshooting)
2. Lihat dokumentasi di folder `docs/`
3. Buat issue di repository untuk bug report
4. Email: [contact-email] untuk support langsung

### Useful Resources:
- OpenCV Documentation: https://docs.opencv.org/
- NumPy User Guide: https://numpy.org/doc/
- Pandas Documentation: https://pandas.pydata.org/docs/

## 📈 Roadmap

### Version 2.0 (Planned):
- [ ] GUI interface dengan tkinter/PyQt
- [ ] Batch processing dari command line
- [ ] Export ke Excel dengan charts
- [ ] Real-time preview ROI selection
- [ ] Custom metric plugins system

### Version 1.5 (Coming Soon):
- [ ] SSIM dan PSNR metrics
- [ ] Statistical analysis summary
- [ ] Image preprocessing options
- [ ] Configuration file support

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ ENT, EME, CII metrics calculation
- ✅ Multi-folder comparison
- ✅ ROI analysis options
- ✅ Parallel processing
- ✅ Comprehensive error handling
- ✅ Progress tracking
- ✅ Developer documentation

---

## 📄 Lisensi

Program ini dibuat untuk keperluan penelitian dan edukasi dalam bidang medical imaging. Silakan gunakan dan modifikasi sesuai kebutuhan dengan menyertakan credit kepada pembuat asli.

**© 2024 - X-Ray Image Quality Analysis Tool**

---

### 🔥 Quick Start

```bash
# 1. Download program
git clone [repo-url]

# 2. Install dependencies  
pip install opencv-python numpy pandas tqdm

# 3. Run program
python main.py

# 4. Follow the interactive prompts
# 5. Check your CSV results!
```

**Happy Analyzing! 🏥📊**