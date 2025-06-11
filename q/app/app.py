import streamlit as st
import pickle
import pandas as pd

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Page settings
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #00ADB5;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .movie-card {
        background-color: #222831;
        padding: 1.2em;
        border-radius: 12px;
        margin: 10px 0;
        color: #eeeeee;
        font-size: 1.2em;
        text-align: center;
        box-shadow: 2px 2px 8px #000;
        transition: 0.3s;
    }
    .movie-card:hover {
        transform: scale(1.02);
        background-color: #393E46;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎬 Movie Recommender</div>", unsafe_allow_html=True)

# Movie selection
selected_movie = st.selectbox("📽️ Choose a movie you like:", movies['title'].values)

# Recommendation logic
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [movies.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Button
if st.button("🚀 Show Recommendations"):
    st.markdown("<h3 style='text-align:center; color:#FFD369;'>🔥 Top Picks for You:</h3>", unsafe_allow_html=True)
    for movie in recommend(selected_movie):
        st.markdown(f"<div class='movie-card'>{movie}</div>", unsafe_allow_html=True)

