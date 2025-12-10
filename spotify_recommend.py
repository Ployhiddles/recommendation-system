import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Columns we expect in your CSV
FEATURE_COLS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "duration_ms",
]

LABEL_COL = "liked"


def prepare_features(df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    return scaler, X


def build_target_vector(feature_means: pd.Series, user_prefs: dict) -> np.ndarray:
    target = feature_means.copy()
    for key, value in user_prefs.items():
        target[key] = value
    return target.values.reshape(1, -1)


def recommend_songs(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    scaler: StandardScaler,
    feature_means: pd.Series,
    user_prefs: dict,
    only_liked: bool,
    n_recs: int,
) -> pd.DataFrame:
    # filter by liked
    mask = pd.Series(True, index=df.index)
    if only_liked and LABEL_COL in df.columns:
        mask &= df[LABEL_COL] == 1

    filtered_df = df[mask].copy()
    if filtered_df.empty:
        return pd.DataFrame()

    filtered_features = features_scaled[mask.values]

    target_raw = build_target_vector(feature_means, user_prefs)
    target_scaled = scaler.transform(target_raw)

    sims = cosine_similarity(target_scaled, filtered_features)[0]
    filtered_df["similarity"] = sims.round(3)

    recs = filtered_df.sort_values("similarity", ascending=False).head(n_recs)
    return recs


def main():
    st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")
    st.title("🎧 Spotify Audio Feature Recommendation System")

    # ✅ LOAD THE DATA FIRST (this was missing and caused NameError)
    df = pd.read_csv("recommendation_df_final.csv")

    # Drop rows with missing feature values
    existing_cols = [col for col in FEATURE_COLS if col in df.columns]
    df = df.dropna(subset=existing_cols)

    if df.empty:
        st.error("All rows are missing required features after cleaning.")
        return

    # Show a sample of the data
    with st.expander("Preview of your data"):
        st.dataframe(df.head())

    # Prepare features
    scaler, features_scaled = prepare_features(df)
    feature_means = df[FEATURE_COLS].mean()

    # ---- Sidebar controls ----
    st.sidebar.header("Your preferences")

    danceability = st.sidebar.slider(
        "Danceability (0 = not danceable, 1 = very danceable)",
        0.0, 1.0, 0.7, 0.05,
    )
    energy = st.sidebar.slider(
        "Energy (0 = calm, 1 = very energetic)",
        0.0, 1.0, 0.8, 0.05,
    )
    valence = st.sidebar.slider(
        "Valence / mood (0 = sad, 1 = happy)",
        0.0, 1.0, 0.7, 0.05,
    )
    acousticness = st.sidebar.slider(
        "Acousticness (0 = electronic, 1 = acoustic)",
        0.0, 1.0, 0.3, 0.05,
    )
    instrumentalness = st.sidebar.slider(
        "Instrumentalness (0 = vocal, 1 = instrumental)",
        0.0, 1.0, 0.2, 0.05,
    )

    tempo = st.sidebar.slider(
        "Tempo (BPM)",
        60.0, 200.0, 120.0, 1.0,
    )
    loudness = st.sidebar.slider(
        "Loudness (dB, -30 = quiet, 0 = loud)",
        -30.0, 0.0, -10.0, 1.0,
    )

    only_liked = st.sidebar.checkbox("Recommend only liked = 1 songs", value=True)

    n_recs = st.sidebar.number_input(
        "Number of recommendations", min_value=1, max_value=50, value=10, step=1
    )

    user_prefs = {
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "tempo": tempo,
        "loudness": loudness,
    }

    if st.sidebar.button("Get recommendations"):
        recs = recommend_songs(
            df=df,
            features_scaled=features_scaled,
            scaler=scaler,
            feature_means=feature_means,
            user_prefs=user_prefs,
            only_liked=only_liked,
            n_recs=n_recs,
        )

        if recs.empty:
            st.warning("No songs matched your filters. Try changing your preferences.")
        else:
            st.subheader("Recommended songs")
            cols_to_show = FEATURE_COLS + ([LABEL_COL] if LABEL_COL in df.columns else []) + ["similarity"]
            st.dataframe(recs[cols_to_show].reset_index(drop=True))


if __name__ == "__main__":
    main()


# ------------- SECOND APP -------------
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Spotify Recommendation system",page_icon="🎧", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json() 

@st.cache_data
def load_file():
    df = pd.read_csv("recommendation_df_final.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    recommendation_df = df.explode("genres")
    return recommendation_df


genre_in_system = ['Jazz', 'Electronic','Dance Pop', 'Hip Hop',  'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock','Tropical', 'Latin Rock','Electropop']
song_characteristics = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence', 'tempo']

recommendation_df = load_file()

def knn_uri(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_df = recommendation_df[(recommendation_df["genres"]==genre) & (recommendation_df["release_year"]>=start_year) & (recommendation_df["release_year"]<=end_year)]
    genre_df = genre_df.sort_values(by='popularity', ascending=False)[:500]

    knn_neigh = NearestNeighbors()
    knn_neigh.fit(genre_df[song_characteristics].to_numpy())

    n_neighbors = knn_neigh.kneighbors([test_feat], n_neighbors=len(genre_df), return_distance=False)[0]

    uris = genre_df.iloc[n_neighbors]["uri"].tolist()
    audios = genre_df.iloc[n_neighbors][song_characteristics].to_numpy()
    return uris, audios


def page():
    st.title("Spotify Song Recommendation System by using Nearest Neighbor Classification")
    st.write("This project is a part from Business Project. This spotify song recommendation by genre and the characteristics of audio,which containing  acousticness, danceability, energy, instrumentalness, valence, tempo. Moreover, this project also apply knn algorithm to generate the hightest k value to find the song with fimilar characteristics")

    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_t9hwygsm.json'
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json)
    df = pd.read_csv("recommendation_df_final.csv")
   
    with st.container():
        col1,col2= st.columns((9,0.5))
        with col1:
            st.caption('Powered by Phuttachat Treerapee')
            st.sidebar.subheader("🏠 Home")
            st.sidebar.subheader("🔎 Search")
            st.sidebar.subheader("📖 Your Library")
            st.sidebar.write("                    ")
            st.sidebar.subheader("➕ Create Playlist")
            st.sidebar.subheader("🎶 Liked Songs")
            st.sidebar.write("____________________")
        
        with col1:
            genre = st.sidebar.selectbox(
                'Genre',genre_in_system)
            st.sidebar.write('your genre selection :', genre,'🌈')
            start_year, end_year = st.sidebar.slider(
                'Song released year',
                2000, 2019, (2000, 2019)
            )
            st.sidebar.write('🐱You selected the year between', start_year, 'and', end_year)


            acousticness = st.sidebar.slider(
                'Acoustic Characteristics',
                0.0, 1.0)
            st.sidebar.write('Acoustic Characteristic Level:', acousticness)
            energy = st.sidebar.slider(
                'Energy Characteristics',
                0.0, 1.0)
            st.sidebar.write('Energy Characteristic Level:', energy)
            danceability = st.sidebar.slider(
                'Danceability Characteristics',
                0.0, 1.0)
            st.sidebar.write('Danceability Characteristic Level:', danceability)
            instrumentalness = st.sidebar.slider(
                'Instrumental Characteristics',
                0.0, 1.0)
            st.sidebar.write('Instrumental Characteristic Level:', instrumentalness)
            tempo = st.sidebar.slider(
                'Tempo Characteristics',
                0.0, 150.0)
            st.sidebar.write('Tempo Characteristic Level:', tempo)
            valence = st.sidebar.slider(
                'Valence Characteristics',
                0.0, 1.0)
            st.sidebar.write('Valence Characteristic Level:', valence)
         
    with st.container():
        col1, col2, col3 = st.columns([3,0.5,3])
        
    
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    uris, audios = knn_uri(genre, start_year, end_year, test_feat)
    songs_number = 10
    
    songs = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        songs.append(track)

    if 'preceding data' not in st.session_state:
        st.session_state['preceding data'] = [genre, start_year, end_year] + test_feat    
    current_inputs = [genre, start_year, end_year] + test_feat    
    if current_inputs != st.session_state['preceding data']:
        if 'song_start' in st.session_state:
            st.session_state['song_start'] = 0
        st.session_state['preceding data'] = current_inputs

    if 'song_start' not in st.session_state:
        st.session_state['song_start'] = 0
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(199, 64, 57);
}
</style>""", unsafe_allow_html=True)
        if st.button("More songs 💿"):
            if st.session_state['song_start'] < len(songs):
                st.session_state['song_start'] += songs_number     

        songs_contem = songs[st.session_state['song_start']: st.session_state['song_start'] + songs_number]
        audios_contem = audios[st.session_state['song_start']: st.session_state['song_start'] + songs_number]     

        if st.session_state['song_start'] < len(songs):
            for i, (track, audio) in enumerate(zip(songs_contem, audios_contem)):
                if i%2==0:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("Data Visualisation🖼️"):
                            df = pd.DataFrame(dict(
                            r=audio[:5],
                            characteristic=song_characteristics[:5]))
                            st.caption('The Polar Chart and Table located below define each characteristic in this song')
                            st.write('Characteristic Level:', df)
                            polar_chart = px.line_polar (df, r='r', theta='characteristic', template = "seaborn", line_close=True, color_discrete_sequence=px.colors.sequential.Blackbody)
                            polar_chart.update_layout(height=260, width=380)
                            st.plotly_chart(polar_chart)
                            
                else:
                    with col1:

                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("Data Visualisation🖼️"):
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                characteristic=song_characteristics[:5]))
                            st.caption('The Polar Chart and Table located below define each characteristic in this song')
                            st.write('Characteristic Level:', df)
                            polar_chart = px.line_polar(df, r='r', theta='characteristic', template = "seaborn",line_close=True, color_discrete_sequence=px.colors.sequential.Blackbody)
                            polar_chart.update_layout(height=260, width=380)
                            st.plotly_chart(polar_chart)
                            
        else:
            with col1:
                st.write("No songs left to recommend")
page()







