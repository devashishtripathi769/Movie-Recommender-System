from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import requests

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.join(BASE_DIR, 'movies.pkl')
similarity_path = os.path.join(BASE_DIR, 'similarity.pkl')

movies = pickle.load(open(movies_path, 'rb'))
similarity = pickle.load(open(similarity_path, 'rb'))

# ---------------- OMDb CONFIG ----------------
OMDB_API_KEY = "be26136c"   # your API key

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# ---------------- POSTER FUNCTION ----------------
def fetch_poster(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("Poster") and data["Poster"] != "N/A":
            return data["Poster"]
    except Exception as e:
        print("Poster fetch error:", e)

    return None

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):
    if not movie:
        return []

    movie = movie.strip()

    # Safety check
    if movie not in movies['title'].values:
        return []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        poster = fetch_poster(title)

        recommendations.append({
            "title": title,
            "poster": poster
        })

    return recommendations

# ---------------- ROUTES ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    movie_list = movies['title'].values
    recommendations = []

    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        print("Selected movie:", selected_movie)  # DEBUG
        recommendations = recommend(selected_movie)
        print("Recommendations:", recommendations)  # DEBUG

    return render_template(
        'index.html',
        movie_list=movie_list,
        recommendations=recommendations
    )

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)





