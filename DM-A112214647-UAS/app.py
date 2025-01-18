#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import spacy
import streamlit as st
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN
from textblob import TextBlob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler

# Function to load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp

# Initialize spaCy model
nlp = load_spacy_model()

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Menghapus URL
    text = text.encode('ascii', 'ignore').decode('ascii')  # Menghapus emoji
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus karakter non-alfabet
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text

# Function to preprocess text with spaCy
def preprocess_with_spacy(text):
    doc = nlp(str(text))
    cleaned_text = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(cleaned_text)

# Function to classify rating into positive or negative
def classify_rating(rating):
    if rating in [3, 4, 5]:
        return 'positive'
    elif rating in [1, 2]:
        return 'negative'
    return 'unknown'

# Load and preprocess data with caching to speed up loading
@st.cache_data
def load_data():
    df = pd.read_csv('data/origin-data/google-play-rev-gen-2.csv', encoding='utf-8')
    df = df.drop(columns=['id', 'title', 'avatar', 'date', 'iso_date', 'response'])
    df['rating_label'] = df['rating'].apply(classify_rating)
    df['cleaned_snippet'] = df['snippet'].apply(clean_text)
    df['cleaned_snippet'] = df['cleaned_snippet'].apply(preprocess_with_spacy)
    return df

# Display dataset preview
def display_data(df):
    st.subheader("Preview Dataset")
    st.write(df.head(20))  # Show only the first 20 rows

# Function to generate word clouds
def generate_wordcloud(df):
    positive_snippets = df[df['rating_label'] == 'positive']['cleaned_snippet']
    negative_snippets = df[df['rating_label'] == 'negative']['cleaned_snippet']
    positive_text = " ".join(positive_snippets.dropna())
    negative_text = " ".join(negative_snippets.dropna())
    
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

    # Display word clouds
    st.subheader('Word Cloud - Positive Reviews')
    st.image(wordcloud_positive.to_array(), use_column_width=True)
    
    st.subheader('Word Cloud - Negative Reviews')
    st.image(wordcloud_negative.to_array(), use_column_width=True)

# Function to train and evaluate model
def train_and_evaluate(df):
    X = df['cleaned_snippet']  # Cleaned text
    y = df['rating_label']  # Sentiment label
    
    # Split the data into training and testing sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Apply ADASYN to handle class imbalance
    adasyn = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_vec, y_train)

    # Train an SVM model
    svm_model = SVC(kernel='rbf', C=1, gamma=0.1, class_weight='balanced', random_state=42)
    svm_model.fit(X_resampled, y_resampled)

    # Predict and evaluate the model
    y_pred = svm_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display results
    st.subheader('Model Evaluation')
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"F1-Score (Weighted): {f1_weighted:.4f}")
    st.text(report)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Display the datasets
def display_datasets():
    df_balanced = pd.read_csv('data/new-changedData/balanced-google-play-rev-gen-2.csv')
    st.subheader('Balanced Google Play Reviews Dataset')
    st.write(df_balanced.head(20))  # Display only the first 20 rows

    # Load and display TF-IDF enhanced dataset
    df_tfidf = pd.read_csv('data/new-changedData/google-play-rev-gen-2-TF_IDF-enhanced.csv')
    st.subheader('TF-IDF Enhanced Google Play Reviews Dataset')
    st.write(df_tfidf.head(20))  # Display only the first 20 rows

# Streamlit UI
def main():
    st.title('A Comparative Analysis of Positive and Negative User Reviews for Genshin Impact on Google Play Store')

    st.sidebar.title('Options')
    options = st.sidebar.radio("Select an option:", ["View Data", "Generate Word Cloud", "Train & Evaluate Model", "TF-IDF Analysis", "View Datasets"])

    df = load_data()

    if options == "View Data":
        display_data(df)

    elif options == "Generate Word Cloud":
        generate_wordcloud(df)

    elif options == "Train & Evaluate Model":
        train_and_evaluate(df)

    elif options == "TF-IDF Analysis":
        tfidf_analysis()

    elif options == "View Datasets":
        display_datasets()

if __name__ == "__main__":
    main()
