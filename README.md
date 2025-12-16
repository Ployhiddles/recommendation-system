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
1. Collect song metadata and audio features from Spotify. The dataset used in this project was obtained from Kaggle and contains songs released between 2000 and 2019.
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
![11](https://github.com/user-attachments/assets/bff8ff77-556f-4970-b4fc-4d008e2b1d51)
![12](https://github.com/user-attachments/assets/e8e52726-a85d-4e71-a385-88b08173b81f)
![13](https://github.com/user-attachments/assets/d8bcc4e6-b830-4830-bb8b-79d3f8727188)




