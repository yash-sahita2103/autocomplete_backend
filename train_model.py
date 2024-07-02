import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Load JSON data
with open('data.json') as f:
    data = json.load(f)

# Extract descriptions, album titles, and song titles
descriptions = []
album_titles = []
song_titles = []
for artist in data:
    for album in artist['albums']:
        descriptions.append(album['description'].strip())
        album_titles.append(album['title'].strip())
        for song in album['songs']:
            song_titles.append(song['title'].strip())

# Combine all texts
all_texts = descriptions + album_titles + song_titles

print(all_texts)

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    return text

# Preprocess all texts
cleaned_texts = [preprocess_text(text) for text in all_texts]

# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform descriptions
X = tfidf_vectorizer.fit_transform(cleaned_texts)

# Initialize k-NN model
knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)

# Save the vectorizer and model to files
joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')
joblib.dump(knn, 'model/knn_model.pkl')
joblib.dump(all_texts, 'model/all_texts.pkl')  # Save the original texts for later use

print("Model and vectorizer saved successfully.")