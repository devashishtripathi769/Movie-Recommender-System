# =========================
# IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd
import ast
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD DATA
# =========================
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# =========================
# MERGE DATA
# =========================
movies = movies.merge(credits, on='title')


# =========================
# SELECT REQUIRED COLUMNS
# =========================
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# =========================
# DROP NULL VALUES
# =========================
movies.dropna(inplace=True)


# =========================
# HELPER FUNCTIONS
# =========================
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []


# =========================
# APPLY FUNCTIONS
# =========================
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)


# =========================
# PROCESS OVERVIEW
# =========================
movies['overview'] = movies['overview'].apply(lambda x: x.split())


# =========================
# REMOVE SPACES
# =========================
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])


# =========================
# CREATE TAGS
# =========================
movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)


# =========================
# FINAL DATAFRAME
# =========================
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# =========================
# VECTORIZATION
# =========================
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()


# =========================
# COSINE SIMILARITY
# =========================
similarity = cosine_similarity(vectors)


# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# =========================
# TEST
# =========================
recommend('Avatar')


# =========================
# SAVE FILES (OPTIONAL)
# =========================
pickle.dump(new_df, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))
print(movies.head())
import pickle
movies_test = pickle.load(open('movies.pkl', 'rb'))
print(movies_test.head())



