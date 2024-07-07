
Overview

The autocomplete_backend project provides a backend service for music title autocomplete using Flask and TF vectorization.


Instalation

Clone the Repository : https://github.com/yash-sahita2103/autocomplete_backend.git
Install the requirements : pip install -r requirements.txt
Train the model : python train_model.py
Hyperpameter Tune (Optional) : python hyperparameter_tuning.py
Start Flask Server : python app.py
-------------------------------------------------------------------------------------------------------------------------------------------------------------

train_model.py

The train_model.py script loads music data from a JSON file, preprocesses textual information by converting to lowercase and removing punctuation, extracts features such as n-grams and substrings from song titles, artist names, and album details, utilizes TF-IDF vectorization to transform textual data into numerical representations, trains a NearestNeighbors model using cosine similarity to identify similar songs based on input queries, and finally saves the trained model, vectorizer, and associated song titles for deployment in a recommendation system.

	Loading and Preprocessing Data:

	Load Data: Loads music data from a JSON file (data.json).
	Preprocess Text: Converts text to lowercase and removes punctuation to standardize text for analysis.
	Feature Extraction:

	Generate Features: Creates n-grams and substrings from song titles, artist names, album titles, song lengths, release years, and genres extracted from the music data.
	Extract Year and Genre: Uses regular expressions to extract year and genre information from album descriptions.
	Vectorization and Model Training:

	Vectorize Text Data: Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical vectors suitable for machine learning.
	Train Nearest Neighbors Model: Trains a NearestNeighbors model using cosine similarity as the distance metric to find songs similar to a given query.
	Saving Model Artifacts:

	Save Model: Saves the trained NearestNeighbors model, TF-IDF vectorizer, and song titles to files (model/model.joblib, model/vectorizer.joblib, model/song_titles.joblib) for later use in the Flask application.
	
	Output:	Prints "Model saved successfully" upon successful completion of model training and saving.

-------------------------------------------------------------------------------------------------------------------------------------------------------------
app.py

The app.py script serves as a backend service for the autocomplete system. It leverages a pre-trained machine learning model and vectorizer to provide autocomplete suggestions based on user input. Users can query song titles, and the system returns relevant suggestions by computing nearest neighbors in a vector space, offering suggestions.

	Initialization and Loading: The script initializes a Flask application and loads a pre-trained model (model.joblib), a vectorizer (vectorizer.joblib), and song titles (song_titles.joblib) used for recommending music based on user queries.

	Endpoint Definition (/autocomplete): This endpoint (GET /autocomplete) handles incoming requests for autocomplete suggestions. It retrieves a query parameter (query), transforms it using the vectorizer, computes nearest neighbors using the model, and filters suggestions based on a distance threshold.

	Function generate_features: This function generates n-grams and substrings from input text, aiding in capturing various features of the input query for similarity matching.

----------------------------------------------------------------------------------------------------------------------------------------------------------------
hyperparameter_tuning.py

The hyperparameter_tuning.py script demonstrates hyperparameter tuning for a autocomplete system using TF-IDF vectorization and cosine similarity. It preprocesses sample music data, computes similarity distances for a query, and retrieves top suggestions based on these distances. This approach allows for fine-tuning parameters like n-gram range and minimum similarity threshold to optimize recommendation accuracy.

	Preprocessing (preprocess_data function): This function initializes a TF-IDF vectorizer (TfidfVectorizer) configured to analyze character-level n-grams (2 to 5 grams). It then fits this vectorizer to the input data, transforming it into a TF-IDF matrix.

	Calculating Distances (calculate_distances function): Given a query string, this function uses a pre-fitted TF-IDF vectorizer to transform the query into a vector representation. It computes cosine similarity between this vector and all vectors in the TF-IDF matrix, resulting in an array of similarity distances.

	Generating Suggestions (get_suggestions function): Using the cosine similarity distances computed in the previous step, this function retrieves the top suggestions that meet a minimum distance threshold (min_distance). It sorts the distances in descending order, selects the top n suggestions, and returns them along with their corresponding distances.
	
