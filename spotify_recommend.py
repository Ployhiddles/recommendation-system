import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Spotify Recommendation System", page_icon="🎧", layout="wide")

# ----------------- CONSTANTS -----------------
genre_in_system = [
    "Jazz", "Electronic", "Dance Pop", "Hip Hop", "K-pop", "Latin",
    "Pop", "Pop Rap", "R&B", "Rock", "Tropical", "Latin Rock", "Electropop"
]

song_characteristics = [
    "acousticness", "danceability", "energy",
    "instrumentalness", "valence", "tempo"
]

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_file():
    df = pd.read_csv("recommendation_df_final.csv")
    # keep your genres logic
    df["genres"] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")]
    )
    recommendation_df = df.explode("genres")
    return recommendation_df

recommendation_df = load_file()

# ----------------- LOTTIE LOADER -----------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ----------------- KNN RECOMMENDER -----------------
def knn_uri(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_df = recommendation_df[
        (recommendation_df["genres"] == genre)
        & (recommendation_df["release_year"] >= start_year)
        & (recommendation_df["release_year"] <= end_year)
    ]
    genre_df = genre_df.sort_values(by="popularity", ascending=False)[:500]

    if genre_df.empty:
        return [], np.array([])

    knn_neigh = NearestNeighbors()
    knn_neigh.fit(genre_df[song_characteristics].to_numpy())

    n_neighbors = knn_neigh.kneighbors(
        [test_feat],
        n_neighbors=len(genre_df),
        return_distance=False
    )[0]

    uris = genre_df.iloc[n_neighbors]["uri"].tolist()
    audios = genre_df.iloc[n_neighbors][song_characteristics].to_numpy()
    return uris, audios

# ----------------- URL → TRACK ID -----------------
def extract_track_id(url_or_uri: str) -> str:
    """
    Accepts:
    - https://open.spotify.com/track/xxxxxx?si=...
    - spotify:track:xxxxxx
    - plain xxxxxx
    Returns the track ID: xxxxxx
    """
    url_or_uri = url_or_uri.strip()

    # Web URL
    if "open.spotify.com" in url_or_uri and "track/" in url_or_uri:
        part = url_or_uri.split("track/")[1]
        track_id = part.split("?")[0]
        return track_id

    # URI form
    if url_or_uri.startswith("spotify:track:"):
        return url_or_uri.split(":")[-1]

    # Assume user already gave just the ID
    return url_or_uri

# ----------------- MAIN PAGE -----------------
def main():
    st.title("Spotify Song Recommendation System")
    st.write(
        "This project recommends songs based on genre and audio characteristics "
        "(acousticness, danceability, energy, instrumentalness, valence, tempo) "
        "and can also generate polar charts for specific tracks from your CSV."
    )

    # Lottie animation
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_t9hwygsm.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json is not None:
        st_lottie(lottie_json)

    # ---------- SIDEBAR: CONTROLS ----------
    st.sidebar.subheader("🏠 Home")
    st.sidebar.subheader("🔎 Search")
    st.sidebar.subheader("📖 Your Library")
    st.sidebar.write("                    ")
    st.sidebar.subheader("➕ Create Playlist")
    st.sidebar.subheader("🎶 Liked Songs")
    st.sidebar.write("____________________")

    st.caption("Powered by Phuttachat Treerapee")

    st.sidebar.header("KNN Recommendation Settings")

    genre = st.sidebar.selectbox("Genre", genre_in_system)
    st.sidebar.write("Your genre selection:", genre, "🌈")

    start_year, end_year = st.sidebar.slider(
        "Song release year",
        2000, 2019, (2000, 2019)
    )
    st.sidebar.write("🐱 You selected the year between", start_year, "and", end_year)

    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5)
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
    instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)
    tempo = st.sidebar.slider("Tempo (BPM)", 0.0, 150.0, 120.0)
    valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)

    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]

    # ---------- LAYOUT ----------
    tab1, tab2 = st.tabs(["🔮 KNN Recommendations", "📊 URL → Polar Chart"])

    # ========== TAB 1: KNN RECOMMENDER ==========
    with tab1:
        st.subheader("KNN-based Song Recommendations")

        uris, audios = knn_uri(genre, start_year, end_year, test_feat)
        songs_number = 10

        if not uris:
            st.warning("No songs found for this genre/year combination.")
        else:
            songs = []
            for uri in uris:
                # If your CSV has full Spotify ID already, this works directly
                embed_id = uri
                # If it has 'spotify:track:ID', you can do:
                # embed_id = uri.split(":")[-1]
                track_iframe = f"""
                <iframe src="https://open.spotify.com/embed/track/{embed_id}"
                        width="260" height="380" frameborder="0"
                        allowtransparency="true" allow="encrypted-media"></iframe>
                """
                songs.append(track_iframe)

            # Session state to paginate songs
            if "preceding_data" not in st.session_state:
                st.session_state["preceding_data"] = [genre, start_year, end_year] + test_feat

            current_inputs = [genre, start_year, end_year] + test_feat
            if current_inputs != st.session_state["preceding_data"]:
                st.session_state["song_start"] = 0
                st.session_state["preceding_data"] = current_inputs

            if "song_start" not in st.session_state:
                st.session_state["song_start"] = 0

            with st.container():
                col1, col2, col3 = st.columns([2, 1, 2])

                st.markdown(
                    """
                    <style>
                    div.stButton > button:first-child {
                        background-color: rgb(199, 64, 57);
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("More songs 💿"):
                    if st.session_state["song_start"] < len(songs):
                        st.session_state["song_start"] += songs_number

                start_idx = st.session_state["song_start"]
                end_idx = start_idx + songs_number
                songs_contem = songs[start_idx:end_idx]
                audios_contem = audios[start_idx:end_idx]

                if start_idx < len(songs):
                    for i, (track_html, audio) in enumerate(zip(songs_contem, audios_contem)):
                        if i % 2 == 0:
                            with col3:
                                components.html(track_html, height=400)
                                with st.expander("Data Visualisation 🖼️"):
                                    df_polar = pd.DataFrame(
                                        dict(
                                            r=audio[:5],
                                            characteristic=song_characteristics[:5],
                                        )
                                    )
                                    st.caption(
                                        "The Polar Chart and table below show each characteristic level for this song."
                                    )
                                    st.write("Characteristic Levels:", df_polar)
                                    polar_chart = px.line_polar(
                                        df_polar,
                                        r="r",
                                        theta="characteristic",
                                        template="seaborn",
                                        line_close=True,
                                        color_discrete_sequence=px.colors.sequential.Blackbody,
                                    )
                                    polar_chart.update_layout(height=260, width=380)
                                    st.plotly_chart(polar_chart, key=f"polar_right_{start_idx}_{i}")
                        else:
                            with col1:
                                components.html(track_html, height=400)
                                with st.expander("Data Visualisation 🖼️"):
                                    df_polar = pd.DataFrame(
                                        dict(
                                            r=audio[:5],
                                            characteristic=song_characteristics[:5],
                                        )
                                    )
                                    st.caption(
                                        "The Polar Chart and table below show each characteristic level for this song."
                                    )
                                    st.write("Characteristic Levels:", df_polar)
                                    polar_chart = px.line_polar(
                                        df_polar,
                                        r="r",
                                        theta="characteristic",
                                        template="seaborn",
                                        line_close=True,
                                        color_discrete_sequence=px.colors.sequential.Blackbody,
                                    )
                                    polar_chart.update_layout(height=260, width=380)
                                    st.plotly_chart(polar_chart, key=f"polar_left_{start_idx}_{i}")
                else:
                    with col1:
                        st.write("No songs left to recommend.")

    # ========== TAB 2: URL → POLAR CHART ==========
    with tab2:
        st.subheader("Generate Polar Chart from Spotify URL / URI / ID")

        track_input = st.text_input(
            "Paste a Spotify track URL / URI / ID:",
            help="Example: https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
        )

        if track_input:
            track_id = extract_track_id(track_input)
            st.write("Detected track ID:", track_id)

            # Try to match in your CSV: uri may be full or just ID
            uri_series = recommendation_df["uri"].astype(str)

            mask_exact = uri_series == track_id
            mask_contains = uri_series.str.contains(track_id, na=False)

            if mask_exact.any():
                row = recommendation_df[mask_exact].iloc[0]
            elif mask_contains.any():
                row = recommendation_df[mask_contains].iloc[0]
            else:
                st.error(
                    "Track not found in the dataset. "
                    "Check that this track exists in recommendation_df_final.csv."
                )
                return

            audio_vals = row[song_characteristics].values

            st.write("Track from CSV:")
            st.write(row[["uri", "genres", "release_year", "popularity"]])

            st.subheader("Audio Feature Values from CSV")
            df_vals = pd.DataFrame(
                {
                    "characteristic": song_characteristics,
                    "value": audio_vals,
                }
            )
            st.write(df_vals)

            polar_df = pd.DataFrame(
                {
                    "characteristic": song_characteristics,
                    "r": audio_vals,
                }
            )

            fig = px.line_polar(
                polar_df,
                r="r",
                theta="characteristic",
                line_close=True,
            )
            fig.update_layout(height=400, width=500)
            st.subheader("Polar Chart")
            st.plotly_chart(fig, key="polar_tab2_chart")

            # Optional Spotify embed
            st.subheader("Spotify Preview")
            embed_url = f"https://open.spotify.com/embed/track/{track_id}"
            st.markdown(
                f'<iframe src="{embed_url}" width="300" height="380" frameborder="0" '
                'allowtransparency="true" allow="encrypted-media"></iframe>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()










