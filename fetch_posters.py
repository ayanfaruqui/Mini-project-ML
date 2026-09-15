"""
Poster Fetcher - Fetches poster paths from TMDB for all movies
Uses TMDB API v3. Run this once to add poster data to the model.
"""
import os
import sys
import pickle
import time
import requests
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
TMDB_IMG_BASE = 'https://image.tmdb.org/t/p/w500'


def fetch_posters_via_api(movie_ids, api_key):
    """Fetch poster paths from TMDB API v3."""
    posters = {}
    total = len(movie_ids)
    success = 0
    batch = 0

    for i, mid in enumerate(movie_ids):
        try:
            url = f'https://api.themoviedb.org/3/movie/{mid}?api_key={api_key}'
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                path = r.json().get('poster_path')
                if path:
                    posters[mid] = path
                    success += 1
            elif r.status_code == 429:
                # Rate limited - wait and retry
                time.sleep(10)
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    path = r.json().get('poster_path')
                    if path:
                        posters[mid] = path
                        success += 1
        except Exception:
            pass

        batch += 1
        if batch >= 35:
            pct = (i + 1) / total * 100
            print(f"   [{pct:5.1f}%] {success}/{i+1} posters fetched...")
            time.sleep(10)  # TMDB rate limit: 40 req / 10 sec
            batch = 0

    return posters, success


def main():
    api_key = os.environ.get('TMDB_API_KEY', '')
    if not api_key:
        print("[!!] TMDB_API_KEY not set!")
        print("     Usage: set TMDB_API_KEY=your_key_here")
        print("     Then run: python fetch_posters.py")
        sys.exit(1)

    # Load existing model
    pkl_path = os.path.join(BASE_DIR, 'movie_dict.pkl')
    if not os.path.exists(pkl_path):
        print("[!!] movie_dict.pkl not found! Run train_model.py first.")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        movies = pickle.load(f)

    print(f"[OK] Loaded {len(movies)} movies")

    # Check if posters already exist
    if 'poster_path' in movies.columns:
        existing = movies['poster_path'].notna().sum()
        print(f"[OK] {existing} movies already have posters")
        if existing > len(movies) * 0.8:
            print("     Most posters already fetched. Skipping.")
            return

    print(f"\n[..] Fetching poster paths from TMDB API...")
    print(f"     This will take ~25 minutes (rate limit: 35 req/10s)")
    print(f"     Total movies: {len(movies)}\n")

    movie_ids = movies['movie_id'].tolist()
    posters, success = fetch_posters_via_api(movie_ids, api_key)

    # Add poster_path column
    movies['poster_path'] = movies['movie_id'].map(posters)

    # Save updated pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(movies, f)

    print(f"\n[OK] Done! {success}/{len(movies)} posters saved to movie_dict.pkl")
    print("     Restart app.py to see posters.")


if __name__ == '__main__':
    main()
