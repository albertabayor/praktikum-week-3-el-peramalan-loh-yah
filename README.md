# praktikum peramalan holt

Repository ini berisi notebook praktikum peramalan time series menggunakan
metode Holt pada dataset `Airline Passengers`. Fokus notebook adalah membangun
alur analisis yang aman untuk laporan: data dimuat, dibersihkan, divalidasi,
divisualisasikan, dibagi menjadi train-test secara time-aware, lalu dipakai
untuk forecasting 15 periode ke depan.

Notebook utama ada di `praktikum_peramalan_holt.ipynb`. Sel kode di dalamnya
sudah dilengkapi dokumentasi bergaya docstring dan komentar singkat agar alur
setiap tahap mudah dipahami seperti dokumentasi fungsi pada bahasa lain.

## gambaran analisis

Notebook ini memakai dataset `Airline Passengers` dari sumber publik berikut:

- `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv`

Karakteristik data:

- frekuensi bulanan,
- periode Januari 1949 sampai Desember 1960,
- total 144 observasi,
- struktur univariat dengan dua kolom utama: `date` dan `value`.

Tahapan analisis di notebook:

1. Import library.
2. Load dataset.
3. Cleaning dan standardisasi kolom.
4. Validasi kualitas data.
5. Visualisasi time series.
6. Identifikasi pola dominan.
7. Train-test split.
8. Fitting model Holt.
9. Forecast 15 periode.
10. Evaluasi error.
11. Interpretasi hasil.
12. Kesimpulan.

## requirement

Anda butuh Python 3 untuk menjalankan notebook. Library yang dipakai:

- `pandas`
- `numpy`
- `matplotlib`
- `requests`
- `statsmodels`
- `ipykernel`

Jika `statsmodels` tidak tersedia, notebook tetap punya fallback implementasi
manual Holt. Namun, kami merekomendasikan tetap menginstalnya agar hasil cocok
dengan implementasi library standar.

## cara menjalankan

Gunakan langkah berikut untuk menyiapkan environment lokal.

1. Buat virtual environment.
2. Aktifkan virtual environment.
3. Install dependency.
4. Jalankan Jupyter Notebook atau buka file notebook di editor yang mendukung
   `.ipynb`.

Contoh perintah:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib requests statsmodels ipykernel
```

Jika Anda memakai Jupyter:

```bash
jupyter notebook praktikum_peramalan_holt.ipynb
```

## validasi yang dilakukan

Notebook memeriksa kualitas data sebelum model dijalankan. Pemeriksaan ini
penting agar forecast tidak dibangun dari data yang bermasalah.

- Jumlah observasi minimal `100`.
- Kolom tanggal berhasil diparse.
- Target tidak memiliki missing value.
- Tidak ada bulan duplikat.
- Urutan waktu ascending.
- Forecast yang dihasilkan tepat `15` periode.
- Metrik evaluasi dapat dihitung tanpa error.

## hasil utama

Hasil eksekusi terakhir notebook di environment lokal menunjukkan ringkasan
berikut:

- dataset: `Airline Passengers`
- jumlah observasi: `144`
- horizon forecast: `15`
- pola dominan: `trend`
- model: `statsmodels.Holt`
- MAE: `66.2480`
- MAPE: `14.7984%`

Interpretasi utamanya adalah data memiliki kecenderungan trend yang cukup
jelas, sehingga Holt masih relevan untuk praktikum ini. Di sisi lain, dataset
ini juga memperlihatkan pola musiman yang tidak dimodelkan secara eksplisit
oleh Holt, sehingga bagian keterbatasan model tetap perlu dibahas di laporan.

<!-- prettier-ignore -->
> [!NOTE]
> Notebook ini disusun untuk kebutuhan praktikum dan laporan akademik.
> Fokus utamanya adalah alur kerja forecasting klasik yang rapi, bukan
> pencarian model paling kompleks atau paling akurat.

## struktur file

Berikut file utama pada repository ini.

- `praktikum_peramalan_holt.ipynb`: notebook analisis lengkap.
- `README.md`: dokumentasi proyek dan panduan menjalankan notebook.

## next steps

Jika Anda ingin mengembangkan praktikum ini lebih lanjut, langkah berikut bisa
menjadi kelanjutan yang wajar:

- tambahkan uji pembanding dengan `Simple Exponential Smoothing`,
- bandingkan hasil Holt dengan `Holt-Winters`,
- simpan visualisasi atau tabel hasil ke file terpisah untuk lampiran laporan.
