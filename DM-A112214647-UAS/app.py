import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load data without preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv('data/origin-data/google-play-rev-gen-2.csv', encoding='utf-8')
    df = df.drop(columns=['id', 'title', 'avatar', 'date', 'iso_date', 'response'])
    return df

# Display dataset
def display_data(df):
    st.subheader("Preview Dataset")
    st.write(df.head())

# Function to generate word clouds
def generate_wordcloud(df):
    positive_snippets = df[df['rating'] >= 3]['snippet']
    negative_snippets = df[df['rating'] < 3]['snippet']
    positive_text = " ".join(positive_snippets.dropna())
    negative_text = " ".join(negative_snippets.dropna())
    
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

    # Display word clouds
    st.subheader('Word Cloud - Positive Reviews')
    st.image(wordcloud_positive.to_array(), use_column_width=True)
    
    st.subheader('Word Cloud - Negative Reviews')
    st.image(wordcloud_negative.to_array(), use_column_width=True)

# Display the datasets
def display_datasets():
    df_original = pd.read_csv('data/origin-data/google-play-rev-gen-2.csv')
    st.subheader('Original Google Play Reviews Dataset')
    st.write(df_original.head(100))

    # Load and display balanced dataset
    df_balanced = pd.read_csv('data/new-changedData/balanced-google-play-rev-gen-2.csv')
    st.subheader('Pre-process result Google Play Reviews Dataset')
    st.write(df_balanced.head(100))

    # Load and display TF-IDF enhanced dataset
    df_tfidf = pd.read_csv('data/new-changedData/google-play-rev-gen-2-TF_IDF-enhanced.csv')
    st.subheader('TF-IDF Enhanced result Google Play Reviews Dataset')
    st.write(df_tfidf.head(100))

# Streamlit UI
def main():
    st.title('A Comparative Analysis of Positive and Negative User Reviews for Genshin Impact on Google Play Store')

    st.sidebar.title('Options')
    options = st.sidebar.radio("Select an option:", ["View Data", "Generate Word Cloud", "View Datasets"])

    df = load_data()

    if options == "View Data":
        display_data(df)

    elif options == "Generate Word Cloud":
        generate_wordcloud(df)

    elif options == "View Datasets":
        display_datasets()

if __name__ == "__main__":
    main()
