import streamlit as st
import pandas as pd
import ast
import json
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the movies and credits dataframes
movies_credits = pd.merge(movies, credits, on='title')

# Extract relevant columns
movies_credits = movies_credits[['movie_id', 'title', 'overview', 'keywords', 'genres', 'cast', 'crew', 'release_date', 'vote_average', 'budget', 'revenue']]

# Remove null values
movies_credits.dropna(inplace=True)

# Function to convert columns to required data types and clean data
def convert_column(column_str, column_type):
    if column_type == 'list':
        items_list = ast.literal_eval(column_str)
        return [item['name'] for item in items_list]
    elif column_type == 'actors':
        cast = json.loads(column_str)
        return [actor['name'] for actor in cast[:5]]
    elif column_type == 'directors':
        crew_list = ast.literal_eval(column_str)
        return [person['name'] for person in crew_list if person['job'] == 'Director']

def convert_movies_credits(movies_credits):
    movies_credits['genres'] = movies_credits['genres'].apply(lambda x: convert_column(x, 'list'))
    movies_credits['keywords'] = movies_credits['keywords'].apply(lambda x: convert_column(x, 'list'))
    movies_credits['cast'] = movies_credits['cast'].apply(lambda x: convert_column(x, 'actors'))
    movies_credits['crew'] = movies_credits['crew'].apply(lambda x: convert_column(x, 'directors'))
    return movies_credits

movies_credits = convert_movies_credits(movies_credits)

# Convert overview to list of sentences
movies_credits['overview'] = movies_credits['overview'].apply(lambda x: x.split())

# Transform columns to remove spaces
def transform_columns(movies_credits, columns):
    for column in columns:
        movies_credits[column] = movies_credits[column].apply(lambda x: [item.replace(' ', '') for item in x])
    return movies_credits

columns_to_transform = ['genres', 'keywords', 'cast', 'crew']
movies_credits = transform_columns(movies_credits, columns_to_transform)

# Concatenate all the columns into a single column (tags)
movies_credits['tags'] = movies_credits['overview'] + movies_credits['genres'] + movies_credits['keywords'] + movies_credits['cast'] + movies_credits['crew']

# Create a new dataframe with movie_id, title, and tags
movies_tags = movies_credits[['movie_id', 'title', 'tags']]
movies_tags['tags'] = movies_tags['tags'].apply(lambda x: ' '.join(x).lower())

# Create a CountVectorizer object and calculate the cosine similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
tags_matrix = cv.fit_transform(movies_tags['tags']).toarray()
similarity_scores = cosine_similarity(tags_matrix)

# API key (hardcoded for the app)
API_KEY = 'cd6d8d77814c6d187e6c131f6a4afd9c'

# Function to fetch movie poster from TMDB API
def get_movie_poster(movie_title):
    movie_details = get_movie_details(movie_title)
    if movie_details and 'poster_path' in movie_details:
        poster_path = movie_details['poster_path']
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    else:
        return "https://via.placeholder.com/150"

# Function to recommend movies
def recommend(movie):
    if movie not in movies_tags['title'].values:
        return [], []  # Movie not found in the dataset
    index = movies_tags[movies_tags['title'] == movie].index[0]
    distances = similarity_scores[index]
    movie_list_sorted = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movie_list_sorted:
        movie_title = movies_tags.iloc[i[0]]['title']
        recommended_movies.append(movie_title)
        recommended_posters.append(get_movie_poster(movie_title))
    
    return recommended_movies, recommended_posters

# Function to fetch movie details from TMDB API
def get_movie_details(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title.replace(' ', '+')}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        movie_data = response.json()
        
        if movie_data['results']:
            movie_id = movie_data['results'][0]['id']
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
            details_response = requests.get(details_url)
            details_response.raise_for_status()
            movie_details = details_response.json()
            return movie_details
        else:
            return None
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Function to display movie details
def display_movie_details(movie_title):
    movie_details = get_movie_details(movie_title)
    
    if movie_details and 'error' not in movie_details:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(get_movie_poster(movie_title))
        with col2:
            st.write(f"**Title**: {movie_details.get('title', 'N/A')}")
            st.write(f"**Overview**: {movie_details.get('overview', 'N/A')}")
            st.write(f"**Release Date**: {movie_details.get('release_date', 'N/A')}")
            st.write(f"**Rating**: {movie_details.get('vote_average', 'N/A')}/10")
            st.write(f"[Official Movie Link](https://www.themoviedb.org/movie/{movie_details.get('id', 'N/A')})")
    else:
        st.error("Movie details not found.")

# Streamlit app
st.title("🎬 Movie Recommendation System")

# Introduction
st.write("""
Welcome to the Movie Recommendation System! 🌟
This app helps you discover movies based on your preferences and provides personalized recommendations.
You can explore recommendations based on a selected movie or filter movies by various criteria.
Use the options below to get started and enjoy your movie exploration experience!
""")

# Add custom CSS for styling
st.markdown("""
<style>
body {
    color: #333;
    background-color: #f0f0f0;
}
.stApp {
    background-color: #f0f0f0;
}
header, .st-c0 {
    background-color: #007BFF;
    color: white;
}
.stButton>button {
    background-color: #007BFF;
    color: white;
    border-radius: 4px;
}
.stButton>button:hover {
    background-color: #0056b3;
}
.stSelectbox>div>div>input, .stTextInput>div>div>input {
    background-color: #e9ecef;
}
.stSelectbox>div>div, .stTextInput>div>div {
    border-radius: 4px;
}
.stRadio>div>div>label, .stSelectbox>div>div>div>label {
    color: #007BFF;
}
.stText {
    color: #333;
}
.stMarkdown {
    color: #333;
}
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
}
.sidebar .sidebar-content .sidebar__header {
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: #007BFF;
}
.stText>div, .stMarkdown>div {
    color: #333;
}
.movie-title {
    color: #0056b3;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Main layout dropdown
st.header("Main Menu")
option = st.selectbox("Choose an option", ["Get Recommendation", "Filtered Movies"], key="main_option")

if 'show_recommended_details' not in st.session_state:
    st.session_state.show_recommended_details = False  # Initialize

if option == "Get Recommendation":
    st.header("🎬 Movie Recommendations")

    # Movie Selection and Recommendation
    selected_movie = st.selectbox("Select a movie to get recommendations", movies_tags['title'].values)

    if selected_movie != "Choose a Movie":
        if st.button("Recommend"):
            recommended_movies, recommended_posters = recommend(selected_movie)
            if recommended_movies:
                st.subheader(f"Recommended Movies based on '{selected_movie}'")
                cols = st.columns(5)
                for i in range(len(recommended_movies)):
                    with cols[i % 5]:
                        st.image(recommended_posters[i], use_column_width=True)
                        st.markdown(f"<div class='movie-title'>{recommended_movies[i]}</div>", unsafe_allow_html=True)
                        if st.button(f"Show Details {i}", key=f"details_recommended_{i}"):
                            st.session_state.selected_movie_details = recommended_movies[i]
                            st.session_state.show_recommended_details = True
            else:
                st.error("No recommendations found. Try selecting a different movie.")

# Display recommended movie details
if 'selected_movie_details' in st.session_state and st.session_state.show_recommended_details:
    st.header("📽️ Movie Details")
    movie_title = st.session_state.selected_movie_details
    display_movie_details(movie_title)
    
    # Reset the detail view
    st.session_state.selected_movie_details = None
    st.session_state.show_recommended_details = False

elif option == "Filtered Movies":
    st.sidebar.title("Filter Options")

    # Sidebar options
    selected_year = st.sidebar.selectbox("Choose Year", ["Choose Year"] + sorted(movies_credits['release_date'].str[:4].unique(), reverse=True))
    selected_genre = st.sidebar.selectbox("Select a Genre", ["All"] + sorted(set(genre for sublist in movies_credits['genres'] for genre in sublist)))
    sort_by = st.sidebar.selectbox("Sort by", ["None", "Rating", "Box Office Collection"])
    rating_threshold = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 5.0)
    search_query = st.sidebar.text_input("Search Movies")

    # Apply filters
    filtered_movies = movies_credits.copy()

    # Apply year filter
    if selected_year != "Choose Year":
        filtered_movies = filtered_movies[filtered_movies['release_date'].str[:4] == selected_year]

    # Apply genre filter
    if selected_genre != "All":
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: selected_genre in x)]

    # Apply rating filter
    filtered_movies = filtered_movies[filtered_movies['vote_average'] >= rating_threshold]

    # Apply search filter
    if search_query:
        filtered_movies = filtered_movies[filtered_movies['title'].str.contains(search_query, case=False)]

    # Apply sort filter
    if sort_by == "Rating":
        filtered_movies = filtered_movies.sort_values(by='vote_average', ascending=False)
    elif sort_by == "Box Office Collection":
        filtered_movies = filtered_movies.sort_values(by='revenue', ascending=False)

    # Limit to top 15 movies based on Box Office Collection
    filtered_movies = filtered_movies.head(15)

    # Display filtered movies
    if not filtered_movies.empty:
        st.header("🎥 Filtered Movies")
        cols = st.columns(5)
        for index, row in filtered_movies.iterrows():
            with cols[index % 5]:
                st.image(get_movie_poster(row['title']), use_column_width=True)
                st.markdown(f"<div class='movie-title'>{row['title']}</div>", unsafe_allow_html=True)
                
                if st.button(f"Show Details {index}", key=f"details_filtered_{index}"):
                    st.session_state.selected_movie_details = row['title']
                    st.session_state.show_recommended_details = True
    else:
        st.write("Use the filters from the sidebar to see movies.")

# Display movie details if selected
if 'selected_movie_details' in st.session_state and st.session_state.show_recommended_details:
    st.header("📽️ Movie Details")
    movie_title = st.session_state.selected_movie_details
    display_movie_details(movie_title)
    
    # Reset the detail view
    st.session_state.show_recommended_details = False
    st.session_state.selected_movie_details = None
