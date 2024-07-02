import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

with open('data.json', 'r', encoding='utf-8') as file:
    music_data = json.load(file)

def extract_features(data):
    features = []
    for artist in data:
        artist_name = artist.get('name', '')
        for album in artist.get('albums', []):
            album_title = album.get('title', '')
            album_description = album.get('description', '')
            for song in album.get('songs', []):
                song_title = song.get('title', '')
                feature = f"{artist_name} {album_title} {album_description} {song_title}"
                features.append((feature, song_title))
    return features

features = extract_features(music_data)
texts, song_titles = zip(*features)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train the nearest neighbors model
model = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(X)

# Save the model and vectorizer
joblib.dump(model, 'model/model.joblib')
joblib.dump(vectorizer, 'model/vectorizer.joblib')
joblib.dump(song_titles, 'model/song_titles.joblib')

print('model saved')
