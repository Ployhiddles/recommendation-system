# Spotify Recommendation System (Streamlit)

A lightweight **Spotify music recommender** built with **Python + Streamlit**.  
Pick a track/artist (or search from Spotify) and the app suggests **similar songs** based on audio features.

🔗 Live demo: https://ployhiddles-recommendation-system-spotify-recommend-k3uhnj.streamlit.app/

---

## Features
- 🔎 Search tracks/artists (Spotify API)
- 🎧 Content-based recommendations using Spotify audio features (e.g., danceability, energy, valence, tempo, etc.)
- 📊 Simple interactive UI with Streamlit
- 🧪 Reproducible local setup (requirements + env variables)

---

## How it works (high level)
1. Fetch track metadata + audio features from Spotify
2. Build a feature vector for the selected track (and/or dataset)
3. Compute similarity (e.g., cosine similarity / nearest neighbors)
4. Return the top-N most similar tracks as recommendations

---

## Tech Stack
- Python
- Streamlit
- Pandas / NumPy
- scikit-learn (similarity / nearest neighbors)
- Spotify Web API (via `spotipy` or direct requests)

---

## Getting Started (Local)

### 1) Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
