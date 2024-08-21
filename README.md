# 🎬 Movie Recommendation System

Welcome to the Movie Recommendation System! This app helps you discover movies based on your preferences and provides personalized recommendations.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommendations-sys.streamlit.app/)

## Description

The Movie Recommendation System is a user-friendly web application built with Streamlit that helps you find movie recommendations based on a selected movie or various filters. By leveraging machine learning techniques and movie metadata, this app provides personalized suggestions and allows you to explore movies through an interactive and visually appealing interface. The system utilizes cosine similarity to recommend movies similar to the one you choose and offers filtering options based on release year, genre, rating, and more. Whether you're looking for new films to watch or want to explore specific genres, this app aims to enhance your movie-watching experience.

## Features

- **Movie Recommendations:** Get personalized movie suggestions based on a selected movie using cosine similarity.
- **Filter Movies:** Apply various filters to discover movies based on release year, genre, rating, and search query.
- **Movie Details:** View detailed information about selected movies, including posters, overviews, release dates, and ratings.
- **Interactive UI:** Enjoy a user-friendly interface with dynamic elements and responsive design.
- **Data Visualization:** See movie recommendations and filtered results in a visually appealing format.
- **External Links:** Access additional resources like movie posters and official movie pages.

## Installation

### Install the Requirements

Create a virtual environment (optional but recommended) and install the necessary packages.

```bash
pip install -r requirements.txt
```
### Running the App

Run the Streamlit App:

```bash
streamlit run streamlit_app.py
```

## Project Structure

- `streamlit_app.py`: The main script for the Streamlit app.
- `tmdb_5000_movies.csv`: Movie dataset.
- `tmdb_5000_credits.csv`: Movie credits dataset.
- `requirements.txt`: List of Python package dependencies.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

Developed with ❤️ by Zain Haidar

- [LinkedIn](https://www.linkedin.com/in/zain-haidar/)
- Email: [contact@zaintheanalyst.com](mailto:contact@zaintheanalyst.com)

Feel free to reach out if you have any questions or feedback!

Note: The app is also hosted on Streamlit Cloud for easy access: [Movie Recommendation System](https://movie-recommendations-sys.streamlit.app/)
