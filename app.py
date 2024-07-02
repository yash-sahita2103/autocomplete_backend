from flask import Flask, request, jsonify
import joblib
import re
import string

app = Flask(__name__)

# Load the saved model and vectorizer
tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
knn = joblib.load('model/knn_model.pkl')
all_texts = joblib.load('model/all_texts.pkl')  # Load the original texts

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    return text

# Function to get suggestions based on user query
def get_suggestions(query, model, vectorizer):
    query_cleaned = preprocess_text(query)
    query_vec = vectorizer.transform([query_cleaned])
    distances, indices = model.kneighbors(query_vec)
    return [all_texts[idx] for idx in indices[0]]

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    suggestions = get_suggestions(query, knn, tfidf_vectorizer)
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)