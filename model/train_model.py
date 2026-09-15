"""
Movie Recommendation System - ML Training Pipeline
===================================================
Hybrid Content-Based + Popularity Recommender using TF-IDF & Cosine Similarity
Dataset: TMDB 5000 Movies
"""

import os
import sys
import ast
import pickle
import time
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TMDB API for fetching poster paths
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')
TMDB_IMG_BASE = 'https://image.tmdb.org/t/p/w500'


def download_dataset():
    """Download TMDB 5000 dataset from a public source."""
    movies_path = os.path.join(BASE_DIR, 'tmdb_5000_movies.csv')

    if os.path.exists(movies_path):
        print("[OK] Dataset file already exists.")
        return movies_path

    print("[>>] Downloading TMDB 5000 dataset...")

    urls = [
        'https://raw.githubusercontent.com/vamshi121/TMDB-5000-Movie-Dataset/master/tmdb_5000_movies.csv',
    ]

    for url in urls:
        try:
            print(f"   Trying: {url[:60]}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(movies_path, 'wb') as f:
                f.write(response.content)
            print("[OK] Dataset downloaded successfully!")
            return movies_path
        except Exception as e:
            print(f"   Failed: {e}")
            continue

    print("[!!] Download failed.")
    print("     Please download the TMDB 5000 dataset manually from Kaggle:")
    print("     https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
    print(f"     Place 'tmdb_5000_movies.csv' in: {BASE_DIR}")
    raise SystemExit(1)


def parse_json_column(text):
    """Safely parse JSON-like string columns from the dataset."""
    try:
        return [item['name'] for item in ast.literal_eval(text)]
    except (ValueError, SyntaxError, TypeError):
        return []


def fetch_poster_path(movie_id):
    """Fetch poster path from TMDB API for a single movie."""
    if not TMDB_API_KEY:
        return None
    try:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}'
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('poster_path', None)
    except Exception:
        pass
    return None


def fetch_all_posters(movie_ids):
    """Batch fetch poster paths for all movies from TMDB API."""
    if not TMDB_API_KEY:
        print("   [!!] No TMDB_API_KEY set. Skipping poster fetch.")
        print("        Set TMDB_API_KEY environment variable and re-train to add posters.")
        return {}

    print(f"   Fetching poster paths for {len(movie_ids)} movies from TMDB...")
    posters = {}
    success_count = 0
    batch_size = 35  # TMDB rate limit: 40 requests per 10 seconds

    for i, movie_id in enumerate(movie_ids):
        poster_path = fetch_poster_path(movie_id)
        if poster_path:
            posters[movie_id] = poster_path
            success_count += 1

        # Rate limiting: pause every batch_size requests
        if (i + 1) % batch_size == 0:
            progress = (i + 1) / len(movie_ids) * 100
            print(f"   [{progress:5.1f}%] Fetched {success_count}/{i+1} posters...")
            time.sleep(10)  # Respect TMDB rate limit

    print(f"   [OK] Fetched {success_count}/{len(movie_ids)} poster paths.")
    return posters


def preprocess_data(movies_path):
    """Load and preprocess the dataset."""
    print("\n[..] Preprocessing data...")

    movies = pd.read_csv(movies_path)
    movies = movies.rename(columns={'id': 'movie_id'})

    movies = movies[[
        'movie_id', 'title', 'overview', 'genres', 'keywords',
        'production_companies', 'vote_average', 'vote_count',
        'popularity', 'release_date'
    ]].copy()

    movies.dropna(subset=['overview'], inplace=True)

    movies['genres'] = movies['genres'].apply(parse_json_column)
    movies['keywords'] = movies['keywords'].apply(parse_json_column)
    movies['production_companies'] = movies['production_companies'].apply(
        lambda x: parse_json_column(x)[:3]
    )

    for col in ['genres', 'keywords', 'production_companies']:
        movies[col] = movies[col].apply(
            lambda lst: [item.replace(' ', '') for item in lst]
        )

    movies['tags'] = (
        movies['overview'].apply(lambda x: str(x).split()) +
        movies['genres'] * 2 +
        movies['keywords'] +
        movies['production_companies']
    )
    movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x).lower())

    vote_counts = movies['vote_count'].fillna(0)
    vote_avg = movies['vote_average'].fillna(0)
    movies['popularity_score'] = vote_avg * np.log1p(vote_counts)

    movies['year'] = pd.to_datetime(
        movies['release_date'], errors='coerce'
    ).dt.year.fillna(0).astype(int)

    movies['genre_list'] = movies['genres']
    movies = movies.reset_index(drop=True)

    print(f"   Processed {len(movies)} movies.")
    return movies


def build_model(movies):
    """Build TF-IDF vectors and cosine similarity matrix."""
    print("\n[..] Building recommendation model...")

    ps = PorterStemmer()
    movies['tags'] = movies['tags'].apply(
        lambda text: ' '.join([ps.stem(word) for word in text.split()])
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(movies['tags'])
    print(f"   TF-IDF matrix shape: {vectors.shape}")

    similarity = cosine_similarity(vectors)
    print(f"   Similarity matrix shape: {similarity.shape}")

    return similarity


def save_model(movies, similarity):
    """Save model artifacts as pickle files."""
    print("\n[..] Saving model artifacts...")

    movie_data = movies[[
        'movie_id', 'title', 'overview', 'genre_list',
        'vote_average', 'year', 'popularity_score', 'poster_path'
    ]].copy()
    movie_data = movie_data.reset_index(drop=True)

    movie_dict_path = os.path.join(BASE_DIR, 'movie_dict.pkl')
    similarity_path = os.path.join(BASE_DIR, 'similarity.pkl')

    with open(movie_dict_path, 'wb') as f:
        pickle.dump(movie_data, f)
    size1 = os.path.getsize(movie_dict_path) / 1e6
    print(f"   [OK] Saved movie_dict.pkl ({size1:.1f} MB)")

    with open(similarity_path, 'wb') as f:
        pickle.dump(similarity, f)
    size2 = os.path.getsize(similarity_path) / 1e6
    print(f"   [OK] Saved similarity.pkl ({size2:.1f} MB)")


def test_recommendations(movies, similarity):
    """Quick test to verify the model works."""
    print("\n[..] Testing recommendations...")

    test_movies = ['The Dark Knight', 'Avatar', 'Titanic']
    for test_movie in test_movies:
        matches = movies[movies['title'] == test_movie]
        if matches.empty:
            continue

        idx = matches.index[0]
        distances = sorted(
            list(enumerate(similarity[idx])),
            reverse=True, key=lambda x: x[1]
        )[1:6]

        recs = [movies.iloc[i[0]]['title'] for i in distances]
        print(f"\n   If you liked '{test_movie}':")
        for i, rec in enumerate(recs, 1):
            print(f"      {i}. {rec}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  Movie Recommendation System - Training Pipeline")
    print("=" * 60)

    if TMDB_API_KEY:
        print(f"  TMDB API Key: {'*' * 4}{TMDB_API_KEY[-4:]} (set)")
    else:
        print("  TMDB API Key: NOT SET (posters will be placeholders)")
        print("  Tip: set TMDB_API_KEY env variable to fetch movie posters")

    # Step 1: Download dataset
    movies_path = download_dataset()

    # Step 2: Preprocess data
    movies = preprocess_data(movies_path)

    # Step 3: Fetch poster paths from TMDB API
    print("\n[..] Fetching movie posters...")
    poster_map = fetch_all_posters(movies['movie_id'].tolist())
    movies['poster_path'] = movies['movie_id'].map(poster_map)
    poster_count = movies['poster_path'].notna().sum()
    print(f"   [OK] {poster_count}/{len(movies)} movies have poster images.")

    # Step 4: Build model
    similarity = build_model(movies)

    # Step 5: Test recommendations
    test_recommendations(movies, similarity)

    # Step 6: Save model
    save_model(movies, similarity)

    print("\n" + "=" * 60)
    print("  [OK] Training complete! Model saved successfully.")
    print("  Run 'python app.py' to start the web server.")
    print("=" * 60)


if __name__ == '__main__':
    main()
