## Nama : Muhammad Faris Assami
## NIM  : A11.2022.14647
## Kelas: A11-4504

# Proyek Klasifikasi Sentimen Ulasan Google Play - Genshin Impact

Repositori ini mengintegrasikan tiga langkah utama untuk mengumpulkan, memproses, dan menganalisis ulasan aplikasi dari Google Play Store. Proyek ini difokuskan pada ulasan aplikasi "Genshin Impact" dengan tujuan untuk mengklasifikasikan sentimen pengguna berdasarkan ulasan yang diberikan, serta untuk mengeksplorasi metode analisis sentimen menggunakan pembelajaran mesin.

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

Berikut adalah daftar pustaka yang diperlukan untuk menjalankan proyek ini:

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
   - Pastikan Anda sudah memiliki file `.env` dengan kunci API SerpApi Anda.
   - Jalankan `serpapi-data.py` untuk mengunduh data ulasan dan menyimpannya dalam file CSV.

### 2. Pra-Pemrosesan Data
   - Jalankan `pre-process-data.ipynb` untuk membersihkan dan mempersiapkan data ulasan.

### 3. Analisis Data
   - Jalankan `analysis.ipynb` untuk melakukan analisis sentimen menggunakan model SVM dan mengevaluasi kinerjanya.

---
