"""
Movie Recommendation System - ML Training Pipeline
===================================================
Hybrid Content-Based + Popularity Recommender using TF-IDF & Cosine Similarity
Dataset: TMDB 5000 Movies
"""

import os
import ast
import pickle
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


def download_dataset():
    """Download TMDB 5000 dataset from a public source."""
    movies_path = os.path.join(BASE_DIR, 'tmdb_5000_movies.csv')

    if os.path.exists(movies_path):
        print("[OK] Dataset file already exists.")
        return movies_path

    print("[>>] Downloading TMDB 5000 dataset...")

    # Try downloading from GitHub mirrors
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


def preprocess_data(movies_path):
    """Load and preprocess the dataset."""
    print("\n[..] Preprocessing data...")

    # Load dataset
    movies = pd.read_csv(movies_path)

    # Rename 'id' to 'movie_id' for consistency
    movies = movies.rename(columns={'id': 'movie_id'})

    # Select relevant columns
    movies = movies[[
        'movie_id', 'title', 'overview', 'genres', 'keywords',
        'production_companies', 'vote_average', 'vote_count',
        'popularity', 'release_date'
    ]].copy()

    # Drop rows with missing overview
    movies.dropna(subset=['overview'], inplace=True)

    # Parse JSON columns
    movies['genres'] = movies['genres'].apply(parse_json_column)
    movies['keywords'] = movies['keywords'].apply(parse_json_column)
    movies['production_companies'] = movies['production_companies'].apply(
        lambda x: parse_json_column(x)[:3]  # Top 3 production companies
    )

    # Remove spaces from multi-word names (so "Science Fiction" -> "ScienceFiction")
    for col in ['genres', 'keywords', 'production_companies']:
        movies[col] = movies[col].apply(
            lambda lst: [item.replace(' ', '') for item in lst]
        )

    # Create tags column by combining all text features
    movies['tags'] = (
        movies['overview'].apply(lambda x: str(x).split()) +
        movies['genres'] * 2 +          # Boost genre weight
        movies['keywords'] +
        movies['production_companies']
    )
    movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x).lower())

    # Calculate popularity score for hybrid ranking
    vote_counts = movies['vote_count'].fillna(0)
    vote_avg = movies['vote_average'].fillna(0)
    movies['popularity_score'] = vote_avg * np.log1p(vote_counts)

    # Extract release year
    movies['year'] = pd.to_datetime(
        movies['release_date'], errors='coerce'
    ).dt.year.fillna(0).astype(int)

    # Store genres for display
    movies['genre_list'] = movies['genres']

    # Reset index
    movies = movies.reset_index(drop=True)

    print(f"   Processed {len(movies)} movies.")
    return movies


def build_model(movies):
    """Build TF-IDF vectors and cosine similarity matrix."""
    print("\n[..] Building recommendation model...")

    # Stemming
    ps = PorterStemmer()
    movies['tags'] = movies['tags'].apply(
        lambda text: ' '.join([ps.stem(word) for word in text.split()])
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(movies['tags'])
    print(f"   TF-IDF matrix shape: {vectors.shape}")

    # Cosine Similarity
    similarity = cosine_similarity(vectors)
    print(f"   Similarity matrix shape: {similarity.shape}")

    return similarity


def save_model(movies, similarity):
    """Save model artifacts as pickle files."""
    print("\n[..] Saving model artifacts...")

    # Prepare movie data for serialization
    movie_data = movies[[
        'movie_id', 'title', 'overview', 'genre_list',
        'vote_average', 'year', 'popularity_score'
    ]].copy()
    movie_data = movie_data.reset_index(drop=True)

    # Save as pickle
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

    # Step 1: Download dataset
    movies_path = download_dataset()

    # Step 2: Preprocess data
    movies = preprocess_data(movies_path)

    # Step 3: Build model
    similarity = build_model(movies)

    # Step 4: Test recommendations
    test_recommendations(movies, similarity)

    # Step 5: Save model
    save_model(movies, similarity)

    print("\n" + "=" * 60)
    print("  [OK] Training complete! Model saved successfully.")
    print("  Run 'python app.py' to start the web server.")
    print("=" * 60)


if __name__ == '__main__':
    main()
