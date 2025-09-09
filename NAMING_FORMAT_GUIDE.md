# PANDUAN FORMAT NAMA FILE GAMBAR

## 🎯 Mengapa Format Nama File Penting?

Program analisis ini membandingkan gambar **berdasarkan nomor urut** di awal nama file. Ini memungkinkan program untuk:

1. **Membandingkan gambar yang sama** dari folder berbeda
2. **Menghitung metrik CII** yang memerlukan perbandingan antar folder
3. **Mengorganisir hasil** secara berurutan

## ✅ Format Yang BENAR

### Template:
```
NOMOR_NAMAFILE.ekstensi
```

### Contoh yang benar:
```
1_xray_dada.jpg
2_rontgen_tangan.png
3_medical_image.tiff
10_scan_kepala.jpg
25_dental_xray.bmp
```

### Penjelasan:
- **NOMOR**: Angka urut (1, 2, 3, dst.) tanpa nol di depan
- **_**: Underscore (garis bawah) sebagai pemisah
- **NAMAFILE**: Nama gambar (bebas, boleh pakai underscore lagi)
- **.ekstensi**: Format file (.jpg, .png, .tiff, dll.)

## ❌ Format Yang SALAH

### Contoh yang salah dan alasannya:

```
❌ xray1.jpg           → Tidak ada nomor di depan
❌ 1-xray.jpg          → Pakai tanda minus (-) bukan underscore (_)
❌ 1 xray.jpg          → Pakai spasi bukan underscore (_)
❌ xray_1.jpg          → Nomor di belakang bukan di depan
❌ 01_xray.jpg         → Ada nol di depan (gunakan 1_xray.jpg)
❌ img1.txt            → Format file tidak didukung
```

## 🔄 Cara Mengubah Nama File

### Metode 1: Manual (untuk file sedikit)
1. Klik kanan pada file
2. Pilih "Rename" atau tekan F2
3. Ubah nama sesuai format: `1_namafile.jpg`

### Metode 2: Batch Rename (untuk banyak file)
1. **Windows**: Gunakan PowerToys PowerRename
2. **Software**: Bulk Rename Utility, Advanced Renamer
3. **Online**: Rename files menggunakan web tools

### Contoh Batch Rename:
```
Dari:  photo1.jpg, photo2.jpg, photo3.jpg
Ke:    1_photo.jpg, 2_photo.jpg, 3_photo.jpg
```

## 🏥 Contoh Penggunaan dalam Analisis Rontgen

### Skenario: Membandingkan gambar sebelum dan sesudah enhancement

**Folder A (Original):**
```
1_chest_original.jpg
2_hand_original.jpg
3_skull_original.jpg
```

**Folder B (Enhanced):**
```
1_chest_enhanced.jpg
2_hand_enhanced.jpg
3_skull_enhanced.jpg
```

**Hasil Analisis:**
- Program akan membandingkan gambar nomor 1 dari kedua folder
- Program akan membandingkan gambar nomor 2 dari kedua folder
- Dan seterusnya...

## 📋 Checklist Sebelum Menjalankan Program

- [ ] Semua file gambar memiliki nomor di depan
- [ ] Menggunakan underscore (_) sebagai pemisah
- [ ] Nomor dimulai dari 1 (bukan 0 atau 01)
- [ ] Format file didukung (.jpg, .png, .tiff, .bmp, .dcm)
- [ ] Tidak ada spasi atau karakter khusus dalam nama
- [ ] File dengan nomor sama ada di setiap folder yang akan dibandingkan

## 🚨 Pesan Error yang Mungkin Muncul

### "TIDAK ADA GAMBAR DENGAN FORMAT YANG BENAR DITEMUKAN"
**Solusi:**
1. Periksa format nama file
2. Pastikan ada underscore setelah nomor
3. Pastikan nomor di awal nama file

### "File dengan format nama salah: [nama file]"
**Solusi:**
1. Ubah nama file sesuai format yang benar
2. Jalankan program kembali setelah perubahan

## 💡 Tips Tambahan

1. **Konsistensi**: Gunakan pola penamaan yang sama untuk semua file
2. **Dokumentasi**: Catat arti dari setiap nomor untuk referensi
3. **Backup**: Buat backup file asli sebelum mengubah nama
4. **Testing**: Coba dengan beberapa file dulu sebelum memproses semua

## 📞 Bantuan

Jika masih mengalami masalah dengan format nama file:
1. Lihat contoh pesan error di program
2. Periksa daftar file yang bermasalah
3. Ikuti panduan perbaikan yang diberikan program