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
import zipfile
import io
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
    credits_path = os.path.join(BASE_DIR, 'tmdb_5000_credits.csv')

    if os.path.exists(movies_path) and os.path.exists(credits_path):
        print("✅ Dataset files already exist.")
        return movies_path, credits_path

    print("📥 Downloading TMDB 5000 dataset...")

    # Try downloading from GitHub mirror
    urls = {
        'movies': 'https://raw.githubusercontent.com/utkarshx27/Movie-Recommender-System/main/tmdb_5000_movies.csv',
        'credits': 'https://raw.githubusercontent.com/utkarshx27/Movie-Recommender-System/main/tmdb_5000_credits.csv'
    }

    try:
        for name, url in urls.items():
            print(f"   Downloading {name}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path = movies_path if name == 'movies' else credits_path
            with open(path, 'wb') as f:
                f.write(response.content)
        print("✅ Dataset downloaded successfully!")
        return movies_path, credits_path
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print("📋 Please download the TMDB 5000 dataset manually from Kaggle:")
        print("   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print(f"   Place 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' in: {BASE_DIR}")
        raise SystemExit(1)


def parse_json_column(text):
    """Safely parse JSON-like string columns from the dataset."""
    try:
        return [item['name'] for item in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []


def get_director(text):
    """Extract director name from the crew column."""
    try:
        for member in ast.literal_eval(text):
            if member.get('job') == 'Director':
                return [member['name']]
    except (ValueError, SyntaxError):
        pass
    return []


def get_top_cast(text, n=3):
    """Extract top N cast members."""
    try:
        return [member['name'] for member in ast.literal_eval(text)[:n]]
    except (ValueError, SyntaxError):
        return []


def preprocess_data(movies_path, credits_path):
    """Load, merge, and preprocess the dataset."""
    print("\n🔧 Preprocessing data...")

    # Load datasets
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Merge on title
    movies = movies.merge(credits, on='title')

    # Select relevant columns
    movies = movies[[
        'movie_id', 'title', 'overview', 'genres', 'keywords',
        'cast', 'crew', 'vote_average', 'vote_count',
        'popularity', 'release_date'
    ]]

    # Drop rows with missing overview
    movies.dropna(subset=['overview'], inplace=True)

    # Parse JSON columns
    movies['genres'] = movies['genres'].apply(parse_json_column)
    movies['keywords'] = movies['keywords'].apply(parse_json_column)
    movies['cast'] = movies['cast'].apply(lambda x: get_top_cast(x, 3))
    movies['director'] = movies['crew'].apply(get_director)

    # Remove spaces from multi-word names (so "Tom Hanks" -> "TomHanks")
    for col in ['genres', 'keywords', 'cast', 'director']:
        movies[col] = movies[col].apply(
            lambda lst: [item.replace(' ', '') for item in lst]
        )

    # Create tags column by combining all text features
    movies['tags'] = (
        movies['overview'].apply(lambda x: x.split()) +
        movies['genres'] +
        movies['keywords'] +
        movies['cast'] +
        movies['director']
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

    # Reset genres back to readable format for display
    movies['genre_list'] = movies['genres'].apply(lambda x: x)

    print(f"   Processed {len(movies)} movies.")
    return movies


def build_model(movies):
    """Build TF-IDF vectors and cosine similarity matrix."""
    print("\n🧠 Building recommendation model...")

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
    print("\n💾 Saving model artifacts...")

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
    print(f"   ✅ Saved movie_dict.pkl ({os.path.getsize(movie_dict_path) / 1e6:.1f} MB)")

    with open(similarity_path, 'wb') as f:
        pickle.dump(similarity, f)
    print(f"   ✅ Saved similarity.pkl ({os.path.getsize(similarity_path) / 1e6:.1f} MB)")


def test_recommendations(movies, similarity):
    """Quick test to verify the model works."""
    print("\n🧪 Testing recommendations...")

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
        print(f"\n   🎬 If you liked '{test_movie}':")
        for i, rec in enumerate(recs, 1):
            print(f"      {i}. {rec}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🎬 Movie Recommendation System - Training Pipeline")
    print("=" * 60)

    # Step 1: Download dataset
    movies_path, credits_path = download_dataset()

    # Step 2: Preprocess data
    movies = preprocess_data(movies_path, credits_path)

    # Step 3: Build model
    similarity = build_model(movies)

    # Step 4: Test recommendations
    test_recommendations(movies, similarity)

    # Step 5: Save model
    save_model(movies, similarity)

    print("\n" + "=" * 60)
    print("✅ Training complete! Model saved successfully.")
    print("   Run 'python app.py' to start the web server.")
    print("=" * 60)


if __name__ == '__main__':
    main()
