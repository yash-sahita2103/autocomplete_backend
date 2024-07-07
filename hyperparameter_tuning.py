import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocess the data
def preprocess_data(data):
    """
    Preprocesses input data using TF-IDF vectorization.

    Args:
    data (list): List of strings to be vectorized.

    Returns:
    tuple: TF-IDF matrix and fitted vectorizer instance.
    """
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    return vectorizer.fit_transform(data), vectorizer

# Calculate distances
def calculate_distances(query, vectorizer, tfidf_matrix):
    """
    Calculates cosine similarity distances between a query and TF-IDF matrix.

    Args:
    query (str): Input query.
    vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer instance.
    tfidf_matrix (sparse matrix): TF-IDF matrix of preprocessed data.

    Returns:
    numpy.array: Array of cosine similarity distances.
    """
    query_vector = vectorizer.transform([query])
    distances = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return distances

# Get suggestions
def get_suggestions(distances, data, n=20, min_distance=0.15):
    """
    Retrieves top suggestions based on cosine similarity distances.

    Args:
    distances (numpy.array): Array of cosine similarity distances.
    data (list): Original data list.
    n (int): Number of suggestions to retrieve (default is 20).
    min_distance (float): Minimum cosine similarity distance threshold.

    Returns:
    list: List of tuples containing suggested strings and their distances.
    """
    indices = np.argsort(-distances)[:n]
    suggestions = [(data[i], distances[i]) for i in indices if distances[i] >= min_distance]
    return suggestions

# Sample data for demonstration
data = [
    "rein raus - rammstein",
    "bloom - radiohead",
    "little by little - radiohead",
    "feral - radiohead",
    "codex - radiohead",
    "mutter - rammstein",
    "wildest dreams - taylor swift",
    "nebel - rammstein",
    "adios - rammstein",
    "zwitter - rammstein",
    "spieluhr - rammstein",
    "feuer frei - rammstein",
    "ich will - rammstein",
    "sonne - rammstein",
    "links 234 - rammstein",
    "rammstein - rammstein",
    "mein herz brennt - rammstein",
    "der meister - rammstein",
    "wollt ihr das bett in flammen sehen - rammstein",
    "asche zu asche - rammstein",
    "seemann - rammstein"
]

# Preprocess data
tfidf_matrix, vectorizer = preprocess_data(data)

# Example query and suggestions
query = "ramm"
distances = calculate_distances(query, vectorizer, tfidf_matrix)
suggestions = get_suggestions(distances, data)

# Print suggestions with distances
for suggestion, distance in suggestions:
    print(f"Suggestion: {suggestion}, Distance: {distance}")

