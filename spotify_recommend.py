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

    # 🔹 1) LOAD YOUR DATA
    df = pd.read_csv("recommendation_df_final.csv")

    # 🔹 2) Drop rows with missing feature values, only for columns that exist
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
            cols_to_show = (
                FEATURE_COLS
                + ([LABEL_COL] if LABEL_COL in df.columns else [])
                + ["similarity"]
            )
            st.dataframe(recs[cols_to_show].reset_index(drop=True))


if __name__ == "__main__":
    main()



