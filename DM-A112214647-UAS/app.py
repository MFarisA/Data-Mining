import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN

# Menampilkan judul aplikasi
st.title('Analisis Sentimen Ulasan Pengguna Game Genshin Impact')

# Memuat data
st.write("Memuat dataset...")
df = pd.read_csv('data/new-changedData/balanced-google-play-rev-gen-2.csv')  # Ganti path sesuai kebutuhan
st.write(df.head())

# Preprocessing Data (misalnya)
st.write("Melakukan pra-pemrosesan pada data...")

# Lakukan analisis model, klasifikasi, dan visualisasi sesuai dengan notebook sebelumnya
X = df['cleaned_snippet']
y = df['rating_label']

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Penyeimbangan data menggunakan ADASYN
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Melatih model SVM
svm_model = SVC(kernel='rbf', C=1, gamma=0.1, class_weight='balanced', random_state=42)
svm_model.fit(X_train_adasyn, y_train_adasyn)

# Evaluasi model
y_pred = svm_model.predict(X_test)
st.write("Evaluasi Model")
st.write(classification_report(y_test, y_pred))
st.write(confusion_matrix(y_test, y_pred))

