from flask import Flask, request, jsonify
import joblib
import json


app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model/model.joblib')
vectorizer = joblib.load('model/vectorizer.joblib')
song_titles = joblib.load('model/song_titles.joblib')

# Load the data to get the Song Artist name
with open('data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)
song_artist_dict = {}
for artist_data in json_data:
    artist_name = artist_data['name']
    for album in artist_data['albums']:
        album_title = album['title']
        for song in album['songs']:
            song_title = song['title']
            song_artist_dict[song_title] = artist_name

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    if not query:
        return jsonify({'suggestions': []})
    
    query_vector = vectorizer.transform([query])
    distances, indices = model.kneighbors(query_vector)
    suggestions = [str(song_titles[index])+' - '+str(song_artist_dict[song_titles[index]]) for index in indices[0]]
    print(suggestions)
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)
