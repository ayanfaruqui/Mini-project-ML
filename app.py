"""
Movie Recommendation System - Flask Web Application
====================================================
Serves the recommendation API and the frontend UI.
"""

import os
import pickle
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load Model Artifacts ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

movie_dict_path = os.path.join(MODEL_DIR, 'movie_dict.pkl')
similarity_path = os.path.join(MODEL_DIR, 'similarity.pkl')

if not os.path.exists(movie_dict_path) or not os.path.exists(similarity_path):
    print("❌ Model files not found!")
    print("   Run 'python model/train_model.py' first to generate model artifacts.")
    raise SystemExit(1)

with open(movie_dict_path, 'rb') as f:
    movies = pickle.load(f)

with open(similarity_path, 'rb') as f:
    similarity = pickle.load(f)

print(f"✅ Loaded {len(movies)} movies and similarity matrix {similarity.shape}")

# ── TMDB API for Posters ──────────────────────────────────────────────
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')
TMDB_IMG_BASE = 'https://image.tmdb.org/t/p/w500'


def fetch_poster(movie_id):
    """Fetch movie poster URL from TMDB API."""
    if not TMDB_API_KEY:
        return None
    try:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}'
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f'{TMDB_IMG_BASE}{poster_path}'
    except Exception:
        pass
    return None


def recommend(movie_title, top_n=5):
    """
    Get movie recommendations using cosine similarity + popularity boost.
    Returns list of recommended movie dicts.
    """
    # Find the movie index
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]

    # Get similarity scores
    sim_scores = list(enumerate(similarity[idx]))

    # Apply hybrid scoring: 0.8 * similarity + 0.2 * normalized_popularity
    max_pop = movies['popularity_score'].max()
    if max_pop > 0:
        hybrid_scores = []
        for i, sim_score in sim_scores:
            pop_score = movies.iloc[i]['popularity_score'] / max_pop
            combined = 0.8 * sim_score + 0.2 * pop_score
            hybrid_scores.append((i, combined))
    else:
        hybrid_scores = sim_scores

    # Sort by hybrid score (exclude the queried movie itself)
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    hybrid_scores = [s for s in hybrid_scores if s[0] != idx][:top_n]

    # Build recommendation list
    recommendations = []
    for movie_idx, score in hybrid_scores:
        movie = movies.iloc[movie_idx]
        poster = fetch_poster(int(movie['movie_id']))

        recommendations.append({
            'title': movie['title'],
            'movie_id': int(movie['movie_id']),
            'overview': movie['overview'][:200] + '...' if len(str(movie['overview'])) > 200 else movie['overview'],
            'genres': movie['genre_list'] if isinstance(movie['genre_list'], list) else [],
            'rating': round(float(movie['vote_average']), 1),
            'year': int(movie['year']) if movie['year'] > 0 else None,
            'poster': poster,
            'score': round(float(score), 3)
        })

    return recommendations


def get_movie_details(movie_title):
    """Get details of the selected movie."""
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return None

    movie = matches.iloc[0]
    poster = fetch_poster(int(movie['movie_id']))

    return {
        'title': movie['title'],
        'movie_id': int(movie['movie_id']),
        'overview': movie['overview'],
        'genres': movie['genre_list'] if isinstance(movie['genre_list'], list) else [],
        'rating': round(float(movie['vote_average']), 1),
        'year': int(movie['year']) if movie['year'] > 0 else None,
        'poster': poster
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/movies')
def get_movies():
    """Return all movie titles for autocomplete."""
    titles = movies['title'].tolist()
    return jsonify({'movies': sorted(titles)})


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """Get recommendations for a given movie."""
    data = request.get_json()
    movie_title = data.get('movie', '')

    if not movie_title:
        return jsonify({'error': 'Movie title is required'}), 400

    # Get selected movie details
    selected = get_movie_details(movie_title)
    if not selected:
        return jsonify({'error': f'Movie "{movie_title}" not found'}), 404

    # Get recommendations
    recs = recommend(movie_title, top_n=5)

    return jsonify({
        'selected': selected,
        'recommendations': recs
    })


@app.route('/api/random')
def random_movies():
    """Return 6 random popular movies for the landing page."""
    popular = movies.nlargest(100, 'popularity_score')
    sample = popular.sample(n=min(6, len(popular)))
    results = []
    for _, movie in sample.iterrows():
        poster = fetch_poster(int(movie['movie_id']))
        results.append({
            'title': movie['title'],
            'movie_id': int(movie['movie_id']),
            'genres': movie['genre_list'] if isinstance(movie['genre_list'], list) else [],
            'rating': round(float(movie['vote_average']), 1),
            'year': int(movie['year']) if movie['year'] > 0 else None,
            'poster': poster
        })
    return jsonify({'movies': results})


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🎬 Movie Recommendation System")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
