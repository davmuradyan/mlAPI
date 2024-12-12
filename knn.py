from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

movies_users = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)
mat_movies = csr_matrix(movies_users.values)

movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movies_users.index)}
index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}

model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=50)
model.fit(mat_movies)


def get_movie_id_by_name_and_year(movie_name, year):
    movie_row = movies[
        (movies['title'].str.contains(movie_name, case=False, na=False)) &
        (movies['title'].str.contains(f"({year})", na=False))
    ]
    print(f"Resolving movie: {movie_name} ({year}), found: {not movie_row.empty}")
    if not movie_row.empty:
        return movie_row.iloc[0]['movieId']
    return None


def recommend_movies(movie_ids, data, n):
    indices = [movie_id_to_index[movie_id] for movie_id in movie_ids if movie_id in movie_id_to_index]

    if not indices:
        return {"error": "No valid movie IDs provided."}

    mean_vector = np.asarray(data[indices].mean(axis=0)).reshape(1, -1)
    distance, rec_indices = model.kneighbors(mean_vector, n_neighbors=50)
    rec_movie_ids = [index_to_movie_id[i] for i in rec_indices[0] if i not in indices]

    if len(rec_movie_ids) < n:
        all_movie_ids = set(index_to_movie_id.values())
        additional_recs = [mid for mid in all_movie_ids if mid not in rec_movie_ids and mid not in movie_ids]
        rec_movie_ids.extend(additional_recs[:n - len(rec_movie_ids)])

    rec_movie_ids = rec_movie_ids[:n]

    recommendations = []
    for mid in rec_movie_ids:
        title = movies[movies['movieId'] == mid]['title'].values[0]
        # Extract year from the title using regex
        import re
        match = re.search(r"(.*) \((\d{4})\)$", title)
        if match:
            name, year = match.groups()
            recommendations.append({"title": name, "year": year})
        else:
            recommendations.append({"title": title, "year": "Unknown"})

    return {"recommendations": recommendations}

# API endpoint for movie recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    print("Received request:", data)
    input_movies = data.get("movies", [])  # Expecting a list of {"name": "", "year": ""} dictionaries
    n = data.get("n", 10)

    if not isinstance(input_movies, list) or not input_movies:
        return jsonify({"error": "Please provide a list of movies with their names and years."}), 400

    if not isinstance(n, int) or n <= 0:
        return jsonify({"error": "Please provide a positive integer for 'n'."}), 400

    movie_ids = []
    for movie in input_movies:
        name = movie.get("name")
        year = movie.get("year")
        if not name or not year:
            return jsonify({"error": "Each movie must have a 'name' and a 'year'."}), 400

        movie_id = get_movie_id_by_name_and_year(name, year)
        if movie_id:
            movie_ids.append(movie_id)

    if not movie_ids:
        return jsonify({"error": "No matching movies found in the dataset."}), 400

    result = recommend_movies(movie_ids, mat_movies, n)
    return jsonify(result)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)