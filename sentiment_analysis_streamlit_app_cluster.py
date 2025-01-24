import streamlit as st
import pandas as pd
import re
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load slang dictionary
def load_slang_dict(path):
    slang_dict = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                slang_dict[key.strip()] = value.strip()
    return slang_dict

slang_dict = load_slang_dict('slang_dict(2).txt')

# Preprocessing function
def preprocess_text(text):
    # Step A.1: Rating converting
    mapping = {'1/5': 'sangat buruk', '2/5': 'buruk', '3/5': 'cukup', '4/5': 'baik', '5/5': 'sangat baik'}
    for key, value in mapping.items():
        text = text.replace(key, value)
    
    # Step A.2: Emoji preprocessing
    text = emoji.demojize(text, delimiters=(" ", ""))
    text = re.sub(r'\s*(\_)\s*', ' ', text)
    
    # Step A.3: Lowercasing, punctuation, numbers, and white space
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.strip().lower()

    # Step C: Standardization (double character)
    text = re.sub(r'([a-z])\1+', r'\1', text)

    # Step C.2: Slang conversion
    words = text.split()
    text = ' '.join([slang_dict.get(word, word) for word in words])
    
    # Step C.3: Remove meaningless words
    del_words = ['rp', 'ny', 'da', 'ah', 'eh']
    text = ' '.join([word for word in text.split() if word not in del_words])

    # Step D: Stemming
    text = stemmer.stem(text)
    
    # Step E: Stopword filtering
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Sentiment extraction
kata_positif = ['baik', 'bagus', 'luar biasa']
kata_negatif = ['buruk', 'jelek', 'parah']
negasi = ["tidak", "bukan", "belum"]
level = ["sangat", "cukup", "kurang", "luar"]

def ekstrak_sentimen(teks):
    kata_kata = teks.lower().split()
    hasil = []
    hasil_ = []

    i = 0
    while i < len(kata_kata):
        word = kata_kata[i]

        # Check for negations
        if word in negasi and i + 1 < len(kata_kata):
            next_word = kata_kata[i + 1]
            if next_word in kata_positif + kata_negatif:
                hasil.append(f"{word} {next_word}")
                i += 2
                hasil_.append(0 if next_word in kata_positif else 1)
                continue

        # Positive/negative word without negation
        if word in kata_positif + kata_negatif:
            hasil.append(word)
            hasil_.append(1 if word in kata_positif else 0)

        i += 1

    return hasil, hasil_

# Ratio and majority functions
def calculate_ratio(entry):
    if len(entry) == 0:
        return 0
    return entry.count(1) / len(entry)

def determine_majority(entry):
    count_1 = entry.count(1)
    count_0 = entry.count(0)
    return 1 if count_1 > count_0 else 0 if count_0 > count_1 else None

# Class mapping
class_denotations = {
    2: "Advocates",
    1: "Detailed Critics",
    3: "Balance Feedback",
    0: "Moderate"
}

# Streamlit UI
st.title("Sentiment Analysis and Class Mapping")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    st.write("Preprocessing text...")
    df['comment_token'] = df['comment'].apply(preprocess_text)

    # Sentiment extraction
    st.write("Extracting sentiment...")
    df[['sentimen word', 'sentimen']] = df['comment_token'].apply(lambda x: pd.Series(ekstrak_sentimen(x)))

    # Calculate ratio and majority
    df['ratio'] = df['sentimen'].apply(calculate_ratio)
    df['sentimen_biner'] = df['sentimen'].apply(determine_majority)

    # Map class_denotation
    st.write("Mapping class denotations...")
    df['class_denotation'] = df['sentimen_biner'].map(class_denotations)

    # Display DataFrame
    st.write("Processed DataFrame:")
    st.dataframe(df)
