# 🎬 CineMatch — AI Movie Recommendation System

A full-stack **Machine Learning** mini project that recommends movies using **Content-Based Filtering** with a hybrid popularity boost. Built with Python, Flask, Scikit-learn, and a stunning dark-themed web UI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

**CineMatch** analyzes 4,800+ movies from the TMDB dataset and uses **TF-IDF Vectorization + Cosine Similarity** to find movies similar to your selection. A popularity boost ensures high-quality recommendations.

### ✨ Features

- 🧠 **ML Model** — Content-based filtering using TF-IDF & Cosine Similarity
- 🎯 **Hybrid Ranking** — Combines content similarity with popularity scoring
- 🔍 **Smart Search** — Live autocomplete with keyboard navigation
- 🎨 **Premium UI** — Dark cinematic theme with glassmorphism effects
- 📱 **Responsive** — Works on desktop, tablet, and mobile
- ⚡ **Fast** — Pre-computed similarity matrix for instant recommendations

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | Scikit-learn (TF-IDF + Cosine Similarity) |
| Backend | Python Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data | TMDB 5000 Movies Dataset |
| Serialization | Pickle |

---

## 🏗️ Architecture

```
User Input → Search Autocomplete → Flask API
                                      ↓
                              Load Pickle Model
                                      ↓
                           Cosine Similarity Lookup
                                      ↓
                            Popularity Re-ranking
                                      ↓
                           Return Top 5 Movies → Render Cards
```

---

## 📁 Project Structure

```
Movie-Recommendation-System/
├── model/
│   └── train_model.py          # ML pipeline: preprocess + train + save
├── static/
│   ├── css/
│   │   └── style.css           # Premium dark-themed UI styles
│   └── js/
│       └── app.js              # Frontend logic
├── templates/
│   └── index.html              # Main web page
├── app.py                      # Flask server + API endpoints
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ayanfaruqui/Mini-project-ML.git
cd Mini-project-ML

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the ML model (one-time, downloads dataset automatically)
python model/train_model.py

# 5. Run the web application
python app.py
```



---

## 🧠 How the ML Model Works

### 1. Data Preprocessing
- Load TMDB 5000 Movies + Credits datasets
- Merge on movie title
- Extract genres, keywords, top 3 cast, director

### 2. Feature Engineering
- Combine all text features into a single **"tags"** column
- Apply stemming (PorterStemmer) and lowercasing

### 3. Vectorization
- **TF-IDF Vectorizer** with 5000 max features
- Converts text tags into numerical vectors

### 4. Similarity Computation
- **Cosine Similarity** between all movie pairs
- Results in a 4806 × 4806 similarity matrix

### 5. Hybrid Ranking
- `score = 0.8 × cosine_similarity + 0.2 × normalized_popularity`
- Popularity = `vote_average × log(1 + vote_count)`

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main web page |
| `/api/movies` | GET | All movie titles (autocomplete) |
| `/api/recommend` | POST | Get 5 recommendations for a movie |
| `/api/random` | GET | 6 random popular movies |

### Example Request

```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"movie": "The Dark Knight"}'
```

---

## 🎨 Screenshots

*Run the app and visit http://localhost:5000 to see the beautiful UI!*

---

## 📊 Dataset

- **Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Size**: ~5000 movies with metadata
- **Features used**: genres, keywords, cast, crew, overview, vote_average, vote_count

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Ayan Faruqui**
- GitHub: [@ayanfaruqui](https://github.com/ayanfaruqui)

---

*Built as a Mini Project for Machine Learning* 🎓
