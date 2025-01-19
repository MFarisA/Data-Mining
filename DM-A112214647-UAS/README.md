## Nama : Muhammad Faris Assami
## NIM  : A11.2022.14647
## Kelas: Data Mining-A11-4504
---
### https://data-mining-9esxueiphlm9yj3tt3uqvf.streamlit.app/
---
# Proyek Analisis Penyebab Rating Rendah dan Tinggi pada Ulasan Pengguna Game Genshin Impact di Google Play Store

**Judul Proyek**: Analisis Penyebab Rating Rendah dan Tinggi pada Ulasan Pengguna Game Genshin Impact di Google Play Store

**Deskripsi Singkat**: Proyek ini bertujuan untuk menganalisis ulasan pengguna game "Genshin Impact" di Google Play Store, dengan fokus pada mengidentifikasi faktor-faktor yang menyebabkan rating rendah (1-2 bintang) dan tinggi (3-5 bintang).

## Ringkasan dan Permasalahan

### Ringkasan:
Eksperimen ini bertujuan untuk mengidentifikasi faktor-faktor yang menyebabkan rating rendah (1-2 bintang) dan tinggi (3-5 bintang) dalam ulasan pengguna game Genshin Impact di Google Play. Dengan menggunakan teknik text mining dan analisis data, proyek ini akan menggali pola-pola umum dalam ulasan untuk memberikan wawasan bagi pengembang terkait area yang perlu diperbaiki dan aspek yang disukai pengguna.

### Permasalahan:
Game mobile, termasuk Genshin Impact, sering menerima ulasan dengan rentang rating yang bervariasi, namun pengembang sering kali kesulitan memahami secara rinci alasan spesifik di balik rating rendah atau tinggi. Ulasan pengguna bisa berisi berbagai topik, mulai dari keluhan teknis hingga pujian tentang gameplay.

### Tujuan:
- Mengidentifikasi faktor-faktor utama yang mempengaruhi pengguna memberikan rating rendah atau tinggi berdasarkan isi ulasan (snippet).
- Memberikan rekomendasi kepada pengembang tentang fitur atau aspek game yang perlu diperbaiki dan apa yang dihargai oleh pengguna.

## Alur Kerja Proyek

Alur kerja proyek ini terbagi menjadi tiga langkah utama, yang dijelaskan secara rinci berikut ini:

### 1. **Pengumpulan Data dengan SerpApi (serpapi-data.py)**

Langkah pertama adalah mengumpulkan data ulasan dari aplikasi "Genshin Impact" di Google Play Store menggunakan API SerpApi. API ini memungkinkan kita untuk mengakses data ulasan secara otomatis dengan mengirimkan permintaan menggunakan kunci API.

#### **Langkah-langkah**:
1. **Memuat Kunci API**:
   - Kunci API disimpan dalam file `.env` dan dipanggil menggunakan pustaka `python-dotenv` untuk menjaga kerahasiaannya.
   - Menggunakan `serpapi.Client` untuk mengonfigurasi klien dengan kunci API yang dimuat.

2. **Mendapatkan Data Ulasan**:
   - Permintaan pencarian dikirim ke SerpApi dengan spesifikasi aplikasi dan jumlah ulasan yang ingin diambil. Dalam hal ini, aplikasi yang digunakan adalah "Genshin Impact", dan jumlah ulasan yang diambil adalah 199.

3. **Menyimpan Data**:
   - Data ulasan yang diterima dari API kemudian disimpan dalam file CSV menggunakan pustaka `pandas`.

#### **Kode Utama**:
```python
import serpapi
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('SERPAPI_KEY')
client = serpapi.Client(api_key=api_key)

results = client.search(
    engine="google_play_product",
    product_id="com.miHoYo.GenshinImpact",
    store="apps",
    all_reviews="true",
    num=199
)

data = results['reviews']
print("total reviews : ", len(results['reviews']))

df = pd.DataFrame(data)
df.to_csv('google-play-rev-gen-2.csv', index=False)
```

### 2. **Pra-Pemrosesan Data (pre-process-data.ipynb)**

Langkah kedua adalah pra-pemrosesan data ulasan yang telah dikumpulkan. Di sini, ulasan dibersihkan dan disiapkan untuk analisis lebih lanjut.

#### **Langkah-langkah**:
1. **Pembersihan Data**:
   - Menghapus URL, emoji, dan karakter non-alfabet.
   - Menormalkan teks dengan mengubah semua huruf menjadi huruf kecil dan menghapus spasi berlebih.

2. **Klasifikasi Rating**:
   - Menggunakan logika untuk mengkategorikan ulasan ke dalam dua kelas: "positif" (untuk rating 3, 4, dan 5) dan "negatif" (untuk rating 1 dan 2).
   
3. **Lemmatization**:
   - Menggunakan pustaka `spaCy` untuk melakukan lemmatization pada teks, yang bertujuan untuk mengubah kata-kata menjadi bentuk dasar mereka.

4. **Menyimpan Data**:
   - Data yang telah diproses disimpan dalam file CSV baru untuk digunakan dalam langkah berikutnya.

#### **Kode Utama**:
```python
import pandas as pd
import spacy
import re

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Membaca dataset dengan pengecekan encoding
df = pd.read_csv('google-play-rev-gen-2.csv', encoding='utf-8')

# Pembersihan teks
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_snippet'] = df['snippet'].apply(clean_text)
df['rating_label'] = df['rating'].apply(lambda rating: 'positive' if rating in [3, 4, 5] else 'negative')

df.to_csv('google-play-rev-gen-2-processed.csv', index=False)
```

### 3. **Analisis Data (analysis.ipynb)**

Langkah terakhir adalah analisis data yang telah diproses. Di sini, kita menerapkan teknik ekstraksi fitur, pelatihan model pembelajaran mesin, dan evaluasi untuk mendapatkan wawasan sentimen dari data ulasan.

#### **Langkah-langkah**:
1. **Ekstraksi Fitur**:
   - Menggunakan TF-IDF untuk mengubah teks menjadi vektor fitur numerik.
   
2. **Penyeimbangan Data**:
   - Menerapkan ADASYN (Adaptive Synthetic Sampling) untuk mengatasi ketidakseimbangan kelas dalam data pelatihan.

3. **Pelatihan Model SVM**:
   - Melatih model Support Vector Machine (SVM) untuk klasifikasi sentimen.

4. **Evaluasi Model**:
   - Menggunakan metrik seperti akurasi, F1-score, dan matriks kebingungan untuk mengevaluasi kinerja model.

#### **Kode Utama**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN
from collections import Counter

# Load the processed data
df = pd.read_csv('google-play-rev-gen-2-processed.csv')

# Ekstraksi fitur
X = df['cleaned_snippet']
y = df['rating_label']

# Membagi data ke dalam training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Penyeimbangan data menggunakan ADASYN
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Melatih model SVM
svm_model = SVC(kernel='rbf', C=1, gamma=0.1, class_weight='balanced', random_state=42)
svm_model.fit(X_train_adasyn, y_train_adasyn)

# Evaluasi model
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## Persyaratan

Berikut adalah daftar pustaka yang diperlukan untuk menjalankan proyek ini. Untuk menginstal semua pustaka secara otomatis, silakan jalankan perintah `pip install -r requirements.txt` di terminal.

```
pandas
scikit-learn
imblearn
spacy
textblob
matplotlib
seaborn
wordcloud
serpapi
python-dotenv
```


## Cara Menjalankan

### 1. Mengumpulkan Data
   - Pastikan sudah memiliki file `.env` dengan kunci API SerpApi.
   - Jalankan `serpapi-data.py` untuk mengunduh data ulasan dan menyimpannya dalam file CSV.

### 2. Pra-Pemrosesan Data
   - Jalankan `pre-process-data.ipynb` untuk membersihkan dan mempersiapkan data ulasan.

### 3. Analisis Data
   - Jalankan `analysis.ipynb` untuk melakukan analisis sentimen menggunakan model SVM dan mengevaluasi kinerjanya.


## Penjelasan Dataset

Dataset yang digunakan dalam proyek ini terdiri dari ulasan pengguna untuk game "Genshin Impact" yang diambil dari Google Play Store. Dataset ini mencakup beberapa informasi penting, seperti:

- **id**: ID unik untuk setiap ulasan.
- **title**: Nama pengguna atau judul ulasan.
- **avatar**: URL gambar avatar pengguna.
- **rating**: Skor rating yang diberikan oleh pengguna (dalam rentang 1-5).
- **snippet**: Isi dari ulasan pengguna, yang berisi opini atau feedback terkait aplikasi.
- **likes**: Jumlah suka (likes) yang diterima ulasan.
- **date**: Tanggal ulasan diposting.
- **iso_date**: Tanggal dalam format ISO 8601 (waktu standar internasional).
- **response**: Respons dari Developer terhadap ulasan (dalam dataset ini sebagian besar kosong).

## EDA dan Proses Features Dataset

1. **Exploratory Data Analysis (EDA)**:
   - Analisis distribusi rating pengguna (rating rendah dan tinggi).
   - Visualisasi dan pembersihan teks ulasan untuk persiapan analisis lebih lanjut.

2. **Proses Features**:
   - Ekstraksi fitur menggunakan teknik TF-IDF untuk mengubah teks ulasan menjadi vektor numerik.
   - Klasifikasi rating berdasarkan teks ulasan (positif dan negatif).

## Proses Learning / Modeling

1. **Modeling**: 
   - Menggunakan model Support Vector Machine (SVM) untuk klasifikasi sentimen ulasan.
   - Model dilatih dengan data yang sudah diproses (dengan penyeimbangan kelas menggunakan ADASYN).

2. **Evaluasi Model**:
   - Menggunakan metrik evaluasi seperti akurasi, F1-score, dan matriks kebingungan untuk menilai kinerja model.

## Performa Model

Model SVM yang diterapkan berhasil memberikan evaluasi yang baik dengan skor akurasi yang memadai dan F1-score yang tinggi, menunjukkan bahwa model mampu membedakan ulasan dengan rating rendah dan tinggi dengan baik. Hasilnya bisa dilihat pada file `pre-process-data.ipynb`.

## Diskusi Hasil dan Kesimpulan

- Hasil model menunjukkan bahwa ulasan dengan rating rendah sering kali mencakup keluhan teknis dan pengalaman buruk dengan gameplay, sementara ulasan dengan rating tinggi lebih fokus pada aspek positif seperti desain grafis, gameplay, dan pengalaman pengguna secara keseluruhan.
- Developer dapat memanfaatkan wawasan ini untuk meningkatkan kualitas fitur yang lebih dihargai pengguna dan memperbaiki area yang sering dikeluhkan.