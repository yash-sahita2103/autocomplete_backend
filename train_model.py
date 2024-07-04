import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import re

# Load data
with open('data.json', 'r', encoding='utf-8') as file:
    music_data = json.load(file)

# Generate n-grams and substrings
def generate_features(text, n):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    substrings = [text[i:i+length] for length in range(1, len(text)+1) for i in range(len(text)-length+1)]
    return ngrams + substrings

# Extract year and genre from description
def extract_year_and_genre(description):
    year = None
    genre = None
    
    # Extract year (assuming it's a four-digit number in the description)
    year_match = re.search(r'\b(19|20)\d{2}\b', description)
    if year_match:
        year = year_match.group(0)
    
    # Extract genre (assuming it's mentioned explicitly; customize as per your data format)
    genre_match = re.search(r'\b(rock|pop|hip-hop|electronic|jazz|country|classical|alternative|trip-hop|metal|grunge|folk|indie)\b', description, re.IGNORECASE)
    if genre_match:
        genre = genre_match.group(0).capitalize()
    
    return year, genre

# Extract features
def extract_features(data, n=2):
    features = []
    for artist in data:
        artist_name = artist.get('name', 'Unknown Artist')
        for album in artist.get('albums', []):
            album_title = album.get('title', '')
            album_description = album.get('description', '')
            release_year, genre = extract_year_and_genre(album_description)
            for song in album.get('songs', []):
                song_title = song.get('title', '')
                song_length = song.get('length', '')
                # Generate context with additional features
                context = f"{song_title} {artist_name} {album_title} {song_length} {release_year or ''} {genre or ''}"
                ngrams_and_substrings = generate_features(context, n)
                for feature in ngrams_and_substrings:
                    features.append((feature, f"{song_title} - {artist_name}"))
                    
    return features

features = extract_features(music_data, n=3)  # Adjust n-gram size if needed
texts, song_titles = zip(*features)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train the nearest neighbors model
model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(X)

# Save the model and vectorizer
joblib.dump(model, 'model/model.joblib')
joblib.dump(vectorizer, 'model/vectorizer.joblib')
joblib.dump(song_titles, 'model/song_titles.joblib')

print('Model saved successfully')
