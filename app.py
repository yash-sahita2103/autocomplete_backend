from flask import Flask, request, jsonify
import joblib
import logging
from flask_cors import CORS  # Import CORS from flask_cors module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and vectorizer
model = joblib.load('model/model.joblib')
vectorizer = joblib.load('model/vectorizer.joblib')
song_titles = joblib.load('model/song_titles.joblib')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    n_neighbors = int(request.args.get('n_neighbors', 10))
    threshold = 0.4  # Hardcoded threshold value for suggestions
    
    # Generate n-grams and substrings for the query
    query_ngrams_and_substrings = generate_features(query, 3)
    query_features = vectorizer.transform(query_ngrams_and_substrings)
    
    distances, indices = model.kneighbors(query_features, n_neighbors=n_neighbors)
    
    # Log the distances, indices, and query for debugging
    logging.debug(f'Query: {query}')
    logging.debug(f'Distances: {distances}')
    logging.debug(f'Indices: {indices}')

    results = {}
    for dist_list, idx_list in zip(distances, indices):
        for dist, idx in zip(dist_list, idx_list):
            if dist < threshold:  # Apply threshold filter
                suggestion = song_titles[idx]
                # Check if suggestion already exists and has a smaller distance
                if suggestion not in results or dist < results[suggestion]:
                    results[suggestion] = dist
    
    # Sort results by distance and convert to list of tuples
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    # Extract only the suggestions from sorted_results
    suggestions = [result[0] for result in sorted_results]

    # Filter out duplicates
    unique_suggestions = []
    seen = set()
    for suggestion in suggestions:
        if suggestion not in seen:
            unique_suggestions.append(suggestion)
            seen.add(suggestion)

    return jsonify(unique_suggestions[:n_neighbors])

def generate_features(text, n):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    substrings = [text[i:i+length] for length in range(1, len(text)+1) for i in range(len(text)-length+1)]
    return ngrams + substrings

if __name__ == '__main__':
    app.run(debug=True)
