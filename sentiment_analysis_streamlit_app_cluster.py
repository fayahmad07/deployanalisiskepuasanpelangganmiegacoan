import streamlit as st
import pandas as pd
import re
import emoji
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

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

slang_dict = load_slang_dict('slang_dict (2).txt')

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

# Sentiment extraction (no longer required for class denotation)
kata_positif = ['baik','ramah','enak','mantap','nyaman','bagus','suka','rekomendasi','puas','lezat','murah','senang','biasa',
                'sempurna','bersih','indah','cantik','keren','cepat','strategis','sesuai','ramai','keren','juara','kekinian',
                'konsisten','efisien','luas','asik','sopan','hangat','dingim','cukup','lumayan','jelas','teliti','gurih','segar','nikmat',
                'lengkap', 'percaya','hemat','pesona','senyum','padan','sahabat','primadona','nikmat']
kata_negatif = ['kurang','kecewa','buruk','mahal','lama','amis','pelan','jelek','buruk','jijih','lamban','lelet','cukup',
                'kotor','antre','payah','parkir','malas','hambar','aneh','lembek','kesal','jorok','lengket','zonk','lelucon',
                'fiktif','ghaib','tipu','benci','sulit','parah','berisik','kacau','mual','salah']

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

# Cluster centroids
centroids = np.array([
    [-0.38000445, -0.21278075,  0.38032496],
    [-0.05463206, -1.5279502 , -2.4590711 ],
    [ 2.01771611,  0.93361986, -0.85334513],
    [ 0.10251419,  0.58638362,  0.36698813]
])

# Streamlit UI
st.title("Sentiment Analysis and Class Mapping")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    st.write("Preprocessing text...")
    df['comment_token'] = df['comment'].apply(preprocess_text)

    # Sentiment extraction (removed this step for class mapping)
    st.write("Extracting sentiment... (skipping sentiment processing as per request)")

    # Calculate Euclidean distances from the comments' features to the cluster centroids
    st.write("Calculating Euclidean distances to centroids...")
    
    # Assuming the comments are represented as vectors in a feature space (e.g., TF-IDF, word2vec, etc.)
    # Here we simulate scaling the comment vectors with StandardScaler
    # This is where you'd scale or embed the actual features from your comments (e.g., embeddings, TF-IDF, etc.)
    from joblib import load
    scaler = StandardScaler()
    # Simulated scaled features (replace with actual data from text vectorization)
    scaler = load('standardscaler.joblib')
    scaled_features = scaler.fit_transform(df[["word_count", "sentimen_sum", "ratio"]])
    # Compute distances
    distances = euclidean_distances(scaled_features, centroids)
    
    # Assign the closest cluster based on the minimum distance
    closest_clusters = np.argmin(distances, axis=1)

    # Map class denotation based on the assigned cluster
    df['class_denotation'] = [class_denotations.get(cluster, 'Unknown') for cluster in closest_clusters]

    # Display DataFrame
    st.write("Processed DataFrame with class denotations:")
    st.dataframe(df)
