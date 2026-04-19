# LAPORAN PRAKTIKUM TEKNIK PERAMALAN
## Metode Holt pada Data Time Series Bulanan

---

**Nama:** Ade Nafil Firmansah
**NIM:** 3012210002
**Mata Kuliah:** Teknik Peramalan
**Metode:** Holt's Double Exponential Smoothing
**Dataset:** Airline Passengers

---

## 1. Pendahuluan

Peramalan (*forecasting*) merupakan proses memperkirakan nilai suatu variabel di masa mendatang berdasarkan data historis yang tersedia. Dalam konteks bisnis dan industri, peramalan digunakan untuk mendukung pengambilan keputusan di berbagai bidang seperti manajemen produksi, pengendalian persediaan, dan perencanaan kapasitas.

Praktikum ini bertujuan untuk:

1. Mengaplikasikan metode peramalan kuantitatif berbasis **Metode Holt** (*Holt's Double Exponential Smoothing*)
2. Mengolah data time series menggunakan Python
3. Menganalisis hasil peramalan serta mengidentifikasi kelebihan dan keterbatasan metode yang digunakan

---

## 2. Landasan Teori

### 2.1 Exponential Smoothing

Exponential Smoothing adalah metode peramalan yang memberikan bobot lebih besar pada data terbaru dibandingkan data yang lebih lama. Bobot tersebut menurun secara eksponensial seiring bertambahnya usia data.

### 2.2 Metode Holt (Double Exponential Smoothing)

Metode Holt merupakan pengembangan dari *Simple Exponential Smoothing* (SES) yang mampu menangkap komponen **level** dan **trend** dalam data. Metode ini cocok digunakan ketika data memiliki kecenderungan naik atau turun secara konsisten.

Persamaan metode Holt adalah sebagai berikut:

**Persamaan Level:**

$$L_t = \alpha \cdot Y_t + (1 - \alpha)(L_{t-1} + T_{t-1})$$

**Persamaan Trend:**

$$T_t = \beta(L_t - L_{t-1}) + (1 - \beta)T_{t-1}$$

**Persamaan Forecast:**

$$\hat{Y}_{t+h} = L_t + h \cdot T_t$$

**Keterangan:**
- $L_t$ = estimasi level pada periode $t$
- $T_t$ = estimasi trend pada periode $t$
- $Y_t$ = nilai aktual pada periode $t$
- $\alpha$ = parameter pemulusan level ($0 < \alpha < 1$)
- $\beta$ = parameter pemulusan trend ($0 < \beta < 1$)
- $h$ = jumlah periode ke depan yang diramalkan

---

## 3. Dataset

### 3.1 Sumber Data

Dataset yang digunakan adalah **Airline Passengers**, yaitu data jumlah penumpang pesawat per bulan yang bersumber dari repositori publik GitHub:

> **Sumber:** `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv`

Dataset ini merupakan salah satu dataset klasik dalam analisis time series dan telah banyak digunakan sebagai benchmark dalam berbagai penelitian peramalan.

### 3.2 Deskripsi Dataset

| Atribut | Keterangan |
|---|---|
| Periode | Januari 1949 – Desember 1960 |
| Frekuensi | Bulanan |
| Jumlah Observasi | 144 periode |
| Variabel | Jumlah penumpang pesawat (ribuan orang) |

### 3.3 Statistik Deskriptif

| Statistik | Nilai |
|---|---|
| Jumlah data | 144 |
| Rata-rata | 280.30 |
| Standar deviasi | 119.97 |
| Nilai minimum | 104 |
| Kuartil 1 (Q1) | 180.00 |
| Median (Q2) | 265.50 |
| Kuartil 3 (Q3) | 360.50 |
| Nilai maksimum | 622 |

---

## 4. Metodologi

### 4.1 Langkah-Langkah Pengerjaan

Berikut adalah alur kerja yang diterapkan dalam praktikum ini:

```
1. Import Library
       ↓
2. Load Data (Airline Passengers)
       ↓
3. Data Cleaning & Standardisasi
       ↓
4. Validasi Kualitas Data
       ↓
5. Visualisasi Time Series
       ↓
6. Identifikasi Pola Data
       ↓
7. Train-Test Split (time-aware)
       ↓
8. Fitting Model Holt
       ↓
9. Forecast 15 Periode
       ↓
10. Evaluasi Error (MAE & MAPE)
       ↓
11. Interpretasi Hasil
```

### 4.2 Pembersihan dan Standardisasi Data

Tahap pembersihan data meliputi:

- Normalisasi nama kolom menjadi `date` dan `value`
- Pembersihan karakter non-numerik pada kolom target
- Parsing tanggal dengan percobaan beberapa format secara otomatis
- Konversi ke granularitas bulanan (awal bulan)
- Pengurutan data secara kronologis
- Penghapusan duplikasi bulan

### 4.3 Validasi Data

Sebelum melanjutkan ke tahap pemodelan, dilakukan validasi dengan kriteria berikut:

| Kriteria Validasi | Status |
|---|---|
| Jumlah observasi ≥ 100 | ✅ Lolos (144 observasi) |
| Tanggal berhasil di-*parse* | ✅ Lolos |
| Tidak ada nilai target yang *missing* | ✅ Lolos |
| Tidak ada duplikasi bulan | ✅ Lolos |
| Urutan waktu ascending | ✅ Lolos |

### 4.4 Identifikasi Pola Data

Berdasarkan heuristik sederhana yang membandingkan rata-rata awal dan akhir seri serta variasi profil bulanan, diperoleh:

| Ukuran | Nilai |
|---|---|
| Trend strength | 349.50 |
| Koefisien variasi | 0.428 |
| Seasonality ratio | 0.339 |
| **Pola dominan** | **Trend** |

Data Airline Passengers menunjukkan kecenderungan **trend** yang cukup jelas, ditandai dengan peningkatan konsisten jumlah penumpang dari tahun 1949 hingga 1960.

### 4.5 Pembagian Data (Train-Test Split)

Pembagian data dilakukan secara *time-aware* untuk menghindari kebocoran data (*data leakage*):

| Bagian | Periode | Jumlah Observasi |
|---|---|---|
| **Train** | Januari 1949 – September 1959 | 129 periode |
| **Test** | Oktober 1959 – Desember 1960 | 15 periode |

---

## 5. Pemodelan

### 5.1 Implementasi Metode Holt

Model Holt diimplementasikan menggunakan library `statsmodels` dengan konfigurasi:

- **Metode:** `statsmodels.tsa.holtwinters.Holt`
- **Inisialisasi:** `initialization_method='estimated'`
- **Optimasi parameter:** otomatis (`optimized=True`)

### 5.2 Parameter Optimal

Setelah proses optimasi, diperoleh parameter terbaik sebagai berikut:

| Parameter | Nilai | Keterangan |
|---|---|---|
| **Alpha (α)** | ≈ 1.0 | Parameter pemulusan level |
| **Beta (β)** | ≈ 0.0 | Parameter pemulusan trend |

Nilai alpha yang mendekati 1 menunjukkan bahwa model memberikan bobot sangat besar pada observasi terbaru untuk memperbarui estimasi level. Nilai beta yang mendekati 0 menunjukkan bahwa estimasi trend bersifat relatif stabil dan tidak berubah terlalu drastis antar periode.

---

## 6. Hasil Peramalan

### 6.1 Tabel Forecast 15 Periode

Berikut adalah hasil forecast untuk 15 periode ke depan (Oktober 1959 – Desember 1960):

| Periode | Nilai Aktual | Nilai Forecast | Error Absolut | Error (%) |
|---|---|---|---|---|
| Oktober 1959 | 407 | 465.74 | 58.74 | 14.43% |
| November 1959 | 362 | 468.48 | 106.48 | 29.42% |
| Desember 1959 | 405 | 471.23 | 66.23 | 16.35% |
| Januari 1960 | 417 | 473.97 | 56.97 | 13.66% |
| Februari 1960 | 391 | 476.71 | 85.71 | 21.92% |
| Maret 1960 | 419 | 479.45 | 60.45 | 14.43% |
| April 1960 | 461 | 482.20 | 21.20 | 4.60% |
| Mei 1960 | 472 | 484.94 | 12.94 | 2.74% |
| Juni 1960 | 535 | 487.68 | 47.32 | 8.84% |
| Juli 1960 | 622 | 490.42 | 131.58 | 21.15% |
| Agustus 1960 | 606 | 493.16 | 112.84 | 18.62% |
| September 1960 | 508 | 495.91 | 12.09 | 2.38% |
| Oktober 1960 | 461 | 498.65 | 37.65 | 8.17% |
| November 1960 | 390 | 501.39 | 111.39 | 28.56% |
| Desember 1960 | 432 | 504.13 | 72.13 | 16.70% |

### 6.2 Visualisasi Hasil Forecast

Plot perbandingan antara data aktual (train dan test) dengan hasil forecast menunjukkan bahwa:

- **Fitted values** pada data train cukup mengikuti pola data aktual
- **Forecast** menghasilkan garis lurus yang merupakan ekstrapolasi linear dari level dan trend terakhir
- Forecast tidak mampu menangkap fluktuasi musiman yang terlihat pada data aktual

---

## 7. Evaluasi

### 7.1 Metrik Evaluasi

| Metrik | Nilai | Interpretasi |
|---|---|---|
| **MAE** (*Mean Absolute Error*) | **66.25** | Rata-rata selisih absolut antara forecast dan aktual |
| **MAPE** (*Mean Absolute Percentage Error*) | **14.80%** | Rata-rata persentase error relatif terhadap nilai aktual |

**Rumus MAE:**

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |Y_t - \hat{Y}_t|$$

**Rumus MAPE:**

$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{Y_t - \hat{Y}_t}{Y_t} \right| \times 100\%$$

### 7.2 Interpretasi Metrik

- **MAE = 66.25** artinya rata-rata selisih antara forecast dan nilai aktual adalah sekitar 66 ribu penumpang per bulan.
- **MAPE = 14.80%** menunjukkan tingkat akurasi yang **cukup moderat**. Sebagai acuan umum, MAPE di bawah 10% dianggap sangat baik, 10–20% dianggap baik, dan di atas 20% dianggap kurang akurat.

Tingkat error yang tergolong "baik" ini masih dapat dikaitkan dengan ketidakmampuan Metode Holt dalam menangkap pola musiman. Pada data Airline Passengers, terdapat pola musiman yang cukup kuat (misalnya lonjakan penumpang di bulan Juli–Agustus setiap tahun), sementara Holt hanya memodelkan level dan trend.

---

## 8. Analisis

### 8.1 Pola Data

Data Airline Passengers memiliki dua karakteristik utama:

1. **Trend positif yang kuat:** Jumlah penumpang meningkat secara konsisten dari 112 ribu pada Januari 1949 menjadi 432 ribu pada Desember 1960, dengan puncak tertinggi mencapai 622 ribu pada Juli 1960. Ini mencerminkan pertumbuhan industri penerbangan komersial yang pesat pada periode tersebut.

2. **Ada indikasi pola musiman:** Terdapat fluktuasi berulang setiap tahun, dengan puncak yang cenderung muncul pada pertengahan tahun dan penurunan pada akhir hingga awal tahun. Namun, berdasarkan heuristik yang digunakan di notebook, pola dominan data tetap diklasifikasikan sebagai **trend**.

### 8.2 Analisis Hasil Forecast

Dari hasil forecast, beberapa observasi penting dapat dicatat:

- **Error terbesar** terjadi pada Juli 1960 (131.58) karena forecast tidak mampu menangkap lonjakan musiman pada bulan puncak.
- **Error terkecil** terjadi pada September 1960 (12.09) karena nilai aktual kebetulan mendekati ekstrapolasi trend linear Holt.
- **Periode November** (baik 1959 maupun 1960) menunjukkan error besar karena pada bulan-bulan off-peak, nilai aktual jauh di bawah garis trend yang diprediksi Holt.

### 8.3 Kelebihan Metode Holt

1. **Sederhana dan interpretatif:** Model hanya memiliki dua parameter (α dan β) yang mudah dijelaskan dan dipahami.
2. **Efisien secara komputasi:** Proses fitting sangat cepat bahkan pada dataset yang besar.
3. **Tepat untuk data bertrend:** Jika data benar-benar hanya memiliki komponen trend tanpa musiman, Holt memberikan hasil yang optimal.
4. **Tidak memerlukan asumsi distribusi:** Berbeda dengan ARIMA, Holt tidak mengasumsikan distribusi tertentu pada residual.

### 8.4 Keterbatasan Metode Holt

1. **Tidak menangkap seasonality:** Ini adalah keterbatasan terbesar untuk dataset Airline Passengers. Holt menghasilkan forecast berupa garis lurus yang tidak mampu merepresentasikan fluktuasi musiman.
2. **Sensitif terhadap perubahan mendadak:** Jika pola data berubah secara tiba-tiba (seperti resesi atau pandemi), model tidak dapat beradaptasi dengan cepat.
3. **Menghasilkan forecast linier:** Forecast jangka panjang selalu berupa garis lurus, yang tidak realistis untuk banyak fenomena bisnis.
4. **Tidak mempertimbangkan faktor eksternal:** Seperti kebijakan harga, kompetisi, atau kondisi ekonomi makro.

---

## 9. Kesimpulan

Berdasarkan seluruh tahapan praktikum yang telah dilakukan, dapat disimpulkan:

1. **Pipeline forecasting** berbasis Metode Holt berhasil dibangun dengan horizon 15 periode ke depan menggunakan Python dan library `statsmodels`.

2. **Dataset Airline Passengers** dengan 144 observasi bulanan (Januari 1949 – Desember 1960) berhasil diproses, divalidasi, dan digunakan sebagai bahan pemodelan. Semua kriteria validasi data lolos.

3. **Pola dominan data** adalah *trend* positif yang kuat, sehingga Metode Holt secara konseptual relevan dan tepat diterapkan pada dataset ini.

4. **Hasil evaluasi** menunjukkan MAE sebesar 66.25 dan MAPE sebesar 14.80%. Tingkat akurasi ini tergolong dalam kategori "baik" berdasarkan standar MAPE umum (10–20%).

5. **Keterbatasan utama** yang ditemukan adalah Metode Holt tidak mampu menangkap indikasi komponen musiman yang terlihat pada data Airline Passengers. Untuk meningkatkan akurasi, dapat dipertimbangkan penggunaan metode **Holt-Winters** yang menambahkan komponen seasonality, atau model **SARIMA** yang secara eksplisit memodelkan musiman dalam kerangka ARIMA.

---

## 10. Daftar Pustaka

1. Holt, C. C. (1957). *Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages*. ONR Research Memorandum 52, Carnegie Institute of Technology.
2. Makridakis, S., Wheelwright, S. C., & Hyndman, R. J. (1998). *Forecasting: Methods and Applications* (3rd ed.). John Wiley & Sons.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. Tersedia di: https://otexts.com/fpp3/
4. Brownlee, J. (2020). *Airline Passengers Dataset*. Machine Learning Mastery. Tersedia di: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
5. Seabold, S., & Perktold, J. (2010). *Statsmodels: Econometric and Statistical Modeling with Python*. Proceedings of the 9th Python in Science Conference.

---

*Laporan ini dibuat sebagai bagian dari tugas Praktikum Teknik Peramalan.*
