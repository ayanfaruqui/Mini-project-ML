"""
Movie Recommendation System - Flask Web Application
====================================================
Serves the recommendation API and the frontend UI.
Poster paths are pre-fetched during training and stored in the pickle file.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load Model Artifacts ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

movie_dict_path = os.path.join(MODEL_DIR, 'movie_dict.pkl')
similarity_path = os.path.join(MODEL_DIR, 'similarity.pkl')

if not os.path.exists(movie_dict_path) or not os.path.exists(similarity_path):
    print("[!!] Model files not found!")
    print("     Run 'python model/train_model.py' first to generate model artifacts.")
    raise SystemExit(1)

with open(movie_dict_path, 'rb') as f:
    movies = pickle.load(f)

with open(similarity_path, 'rb') as f:
    similarity = pickle.load(f)

# ── TMDB Image Base URL ───────────────────────────────────────────────
TMDB_IMG_BASE = 'https://image.tmdb.org/t/p/w500'

# Check if poster_path column exists (added during training with TMDB_API_KEY)
has_posters = 'poster_path' in movies.columns
poster_count = movies['poster_path'].notna().sum() if has_posters else 0
print(f"[OK] Loaded {len(movies)} movies, similarity matrix {similarity.shape}")
print(f"[OK] Posters available: {poster_count}/{len(movies)}")


def get_poster_url(movie):
    """Get poster URL from pre-stored poster_path."""
    if has_posters:
        poster_path = movie.get('poster_path', None)
        if pd.notna(poster_path) and poster_path:
            return f'{TMDB_IMG_BASE}{poster_path}'
    return None


def recommend(movie_title, top_n=5):
    """
    Get movie recommendations using cosine similarity + popularity boost.
    Returns list of recommended movie dicts.
    """
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]
    sim_scores = list(enumerate(similarity[idx]))

    # Hybrid scoring: 0.8 * similarity + 0.2 * normalized_popularity
    max_pop = movies['popularity_score'].max()
    if max_pop > 0:
        hybrid_scores = []
        for i, sim_score in sim_scores:
            pop_score = movies.iloc[i]['popularity_score'] / max_pop
            combined = 0.8 * sim_score + 0.2 * pop_score
            hybrid_scores.append((i, combined))
    else:
        hybrid_scores = sim_scores

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    hybrid_scores = [s for s in hybrid_scores if s[0] != idx][:top_n]

    recommendations = []
    for movie_idx, score in hybrid_scores:
        movie = movies.iloc[movie_idx]
        recommendations.append({
            'title': movie['title'],
            'movie_id': int(movie['movie_id']),
            'overview': movie['overview'][:200] + '...' if len(str(movie['overview'])) > 200 else movie['overview'],
            'genres': movie['genre_list'] if isinstance(movie['genre_list'], list) else [],
            'rating': round(float(movie['vote_average']), 1),
            'year': int(movie['year']) if movie['year'] > 0 else None,
            'poster': get_poster_url(movie),
            'score': round(float(score), 3)
        })

    return recommendations


def get_movie_details(movie_title):
    """Get details of the selected movie."""
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return None

    movie = matches.iloc[0]
    return {
        'title': movie['title'],
        'movie_id': int(movie['movie_id']),
        'overview': movie['overview'],
        'genres': movie['genre_list'] if isinstance(movie['genre_list'], list) else [],
        'rating': round(float(movie['vote_average']), 1),
        'year': int(movie['year']) if movie['year'] > 0 else None,
        'poster': get_poster_url(movie)
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

    selected = get_movie_details(movie_title)
    if not selected:
        return jsonify({'error': f'Movie "{movie_title}" not found'}), 404

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
        results.append({
            'title': movie['title'],
            'movie_id': int(movie['movie_id']),
            'genres': movie['genre_list'] if isinstance(movie['genre_list'], list) else [],
            'rating': round(float(movie['vote_average']), 1),
            'year': int(movie['year']) if movie['year'] > 0 else None,
            'poster': get_poster_url(movie)
        })
    return jsonify({'movies': results})


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    print("\n  Movie Recommendation System")
    print(f"  Open http://localhost:{port} in your browser\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
