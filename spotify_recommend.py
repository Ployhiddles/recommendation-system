import streamlit as st
import pandas as pd
import plotly.express as px

# 👇 make sure this matches your existing list
song_characteristics = ['acousticness', 'danceability', 'energy',
                        'instrumentalness', 'valence', 'tempo']

# 👇 load your data (or reuse recommendation_df from your app)
@st.cache_data
def load_file():
    df = pd.read_csv("recommendation_df_final.csv")
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")]
    )
    recommendation_df = df.explode("genres")
    return recommendation_df

recommendation_df = load_file()


def extract_track_id(url_or_uri: str) -> str:
    """
    Accepts:
    - https://open.spotify.com/track/xxxxxx?si=...
    - spotify:track:xxxxxx
    - plain xxxxxx
    Returns the track ID: xxxxxx
    """
    url_or_uri = url_or_uri.strip()

    if "open.spotify.com" in url_or_uri and "track/" in url_or_uri:
        part = url_or_uri.split("track/")[1]
        track_id = part.split("?")[0]
        return track_id

    if url_or_uri.startswith("spotify:track:"):
        return url_or_uri.split(":")[-1]

    # assume user already gave just the ID
    return url_or_uri


st.title("Spotify Track Polar Chart from URL")

track_input = st.text_input(
    "Paste a Spotify track URL / URI / ID:",
    help="Example: https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC"
)

if track_input:
    track_id = extract_track_id(track_input)
    st.write("Detected track ID:", track_id)

    # your CSV may store full URI or ID, so we use contains()
    mask = recommendation_df["uri"].astype(str).str.contains(track_id, na=False)

    if not mask.any():
        st.error("Track not found in dataset. Check that this track exists in recommendation_df_final.csv.")
    else:
        row = recommendation_df[mask].iloc[0]
        audio_vals = row[song_characteristics].values

        st.subheader("Audio feature values")
        st.write(pd.DataFrame({
            "characteristic": song_characteristics,
            "value": audio_vals
        }))

        # build polar chart
        polar_df = pd.DataFrame({
            "characteristic": song_characteristics,
            "r": audio_vals
        })

        fig = px.line_polar(
            polar_df,
            r="r",
            theta="characteristic",
            line_close=True
        )
        fig.update_layout(height=400, width=500)
        st.subheader("Audio Feature Polar Chart")
        st.plotly_chart(fig)

        # optional: show playable embed
        embed_url = f"https://open.spotify.com/embed/track/{track_id}"
        st.markdown(
            f'<iframe src="{embed_url}" width="300" height="380" frameborder="0" '
            'allowtransparency="true" allow="encrypted-media"></iframe>',
            unsafe_allow_html=True
        )








