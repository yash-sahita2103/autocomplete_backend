import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import re
import string

# Load data from JSON file
with open('data.json', 'r', encoding='utf-8') as file:
    music_data = json.load(file)

# Preprocess text: lowercase, remove punctuation
def preprocess_text(text):
    """
    Preprocesses text by converting to lowercase and removing punctuation.

    Args:
    text (str): Input text to be processed.

    Returns:
    str: Processed text with lowercase and no punctuation.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Generate n-grams and substrings for text
def generate_features(text, n):
    """
    Generates n-grams and substrings for a given text.

    Args:
    text (str): Input text.
    n (int): Size of n-grams to generate.

    Returns:
    list: List of n-grams and substrings.
    """
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    substrings = [text[i:i+length] for length in range(1, len(text)+1) for i in range(len(text)-length+1)]
    return ngrams + substrings

# Extract year and genre from description
def extract_year_and_genre(description):
    """
    Extracts year and genre from a description text.

    Args:
    description (str): Description text possibly containing year and genre information.

    Returns:
    tuple: Extracted year (str) and genre (str).
    """
    year = None
    genre = None
    
    # Extract year (assuming it's a four-digit number in the description)
    year_match = re.search(r'\b(19|20)\d{2}\b', description)
    if year_match:
        year = year_match.group(0)
    
    # Extract genre (assuming it's mentioned explicitly)
    genre_match = re.search(r'\b(rock|pop|hip-hop|electronic|jazz|country|classical|alternative|trip-hop|metal|grunge|folk|indie)\b', description, re.IGNORECASE)
    if genre_match:
        genre = genre_match.group(0).capitalize()
    
    return year, genre

# Extract features from music data
def extract_features(data, n=2):
    """
    Extracts features from music data including song titles, artist names, album titles, and more.

    Args:
    data (list): Music data in JSON format.
    n (int): Size of n-grams to generate (default is 2).

    Returns:
    list: List of tuples containing features and associated song information.
    """
    features = []
    for artist in data:
        artist_name = preprocess_text(artist.get('name', 'Unknown Artist'))
        for album in artist.get('albums', []):
            album_title = preprocess_text(album.get('title', ''))
            album_description = preprocess_text(album.get('description', ''))
            release_year, genre = extract_year_and_genre(album_description)
            for song in album.get('songs', []):
                song_title = preprocess_text(song.get('title', ''))
                song_length = preprocess_text(song.get('length', ''))
                # Generate context with additional features
                context = f"{song_title} {artist_name} {album_title} {song_length} {release_year or ''} {genre or ''}"
                ngrams_and_substrings = generate_features(context, n)
                for feature in ngrams_and_substrings:
                    features.append((feature, f"{song_title} - {artist_name}"))
                    
    return features

# Extract features from music data with n-gram size of 3
features = extract_features(music_data, n=3)
texts, song_titles = zip(*features)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, max_features=5000, analyzer='char_wb', ngram_range=(2, 5))
X = vectorizer.fit_transform(texts)

# Train the nearest neighbors model using cosine similarity metric
model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(X)

# Save the model, vectorizer, and song titles to disk
joblib.dump(model, 'model/model.joblib')
joblib.dump(vectorizer, 'model/vectorizer.joblib')
joblib.dump(song_titles, 'model/song_titles.joblib')

print('Model saved successfully')