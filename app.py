import streamlit as st
import pandas as pd
import os
import sys

# Ensure project directories are in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.data_loader import DataLoader
from src.recommender import MusicRecommender
from src.spotify_api import SpotifyAPI
from src.utils import setup_logger

logger = setup_logger("app")

# Page Layout & Config
st.set_page_config(
    page_title="VibeSync - Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Styling (Modern Dark Mode with glassmorphism effects)
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Inter', sans-serif;
    }
    .stAppHeader {
        background-color: transparent;
    }
    div[data-testid="stSidebar"] {
        background-color: #1f2833;
        border-right: 1px solid #1f2833;
    }
    h1, h2, h3 {
        color: #66fcf1;
        font-weight: 700;
    }
    /* Song Card Design */
    .song-card {
        background-color: #1f2833;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        border: 1px solid #2f3b4c;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        height: 380px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .song-card:hover {
        transform: translateY(-5px);
        border-color: #66fcf1;
        box-shadow: 0 10px 15px rgba(102, 252, 241, 0.2);
    }
    .song-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-artist {
        color: #66fcf1;
        font-size: 14px;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-meta {
        color: #a4b3c6;
        font-size: 12px;
        margin-bottom: 6px;
    }
    .song-explanation {
        background-color: rgba(102, 252, 241, 0.1);
        border-left: 3px solid #66fcf1;
        padding: 6px;
        border-radius: 4px;
        font-size: 11px;
        color: #66fcf1;
        margin-top: auto;
    }
    /* Buttons and Links */
    .spotify-btn {
        background-color: #1db954;
        color: white !important;
        text-decoration: none;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin-top: 8px;
    }
    .spotify-btn:hover {
        background-color: #1ed760;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Engine and Services
@st.cache_resource
def get_recommender():
    recommender = MusicRecommender()
    recommender.initialize()
    return recommender

@st.cache_resource
def get_spotify_api():
    return SpotifyAPI()

try:
    recommender = get_recommender()
    spotify_api = get_spotify_api()
    df = recommender.df
except Exception as e:
    st.error(f"Critical error initializing recommender engine: {e}")
    st.stop()

# Initialize Session States
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# Sidebar Navigation and Filters
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1614613535308-eb5fbd3d2c17?w=300&auto=format&fit=crop&q=60", use_column_width=True)
    st.title("🎧 VibeSync Controls")
    
    st.subheader("Filters")
    
    # Filter by Language if it exists
    available_languages = ["All"]
    if "language" in df.columns:
        available_languages += sorted(df["language"].dropna().unique().tolist())
    selected_language = st.selectbox("Select Language:", available_languages)
    
    # Filter by Artist if it exists
    available_artists = ["All"]
    available_artists += sorted(df["artist_name"].dropna().unique().tolist())
    selected_artist = st.selectbox("Filter by Artist Name:", available_artists)
    
    st.markdown("---")
    st.markdown("### 🏆 System Statistics")
    st.metric("Total Songs Pool", len(df))
    if "popularity" in df.columns:
        st.metric("Avg Popularity Score", f"{df['popularity'].mean():.1f}/100")

# Apply filters to dataset for dropdown autocomplete search list
filtered_df = df.copy()
if selected_language != "All":
    filtered_df = filtered_df[filtered_df["language"] == selected_language]
if selected_artist != "All":
    filtered_df = filtered_df[filtered_df["artist_name"] == selected_artist]

# App header layout
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🎵 VibeSync - Content-Based Music Recommendation Engine")
    st.markdown(
        "*Discover similar tracks instantly based on metadata, tempo matching, and acoustic features.*"
    )

# Autocomplete Song Search
if not filtered_df.empty:
    # Build track suggestions labels
    track_suggestions = [
        f"{row['track_name']} - {row['artist_name']}" for _, row in filtered_df.iterrows()
    ]
    selected_song_label = st.selectbox(
        "🔎 Search or Select a song to generate recommendations:",
        track_suggestions
    )
    
    # Parse back the selected song and artist name
    selected_song_name = selected_song_label.split(" - ")[0].strip()
    selected_artist_name = selected_song_label.split(" - ")[1].strip()
    
    selected_index = recommender.get_track_index(selected_song_name, selected_artist_name)
else:
    st.warning("No songs match the selected filters. Please clear filters in the sidebar.")
    selected_index = None

# Main Operations
if selected_index is not None:
    current_song = df.iloc[selected_index]
    
    # Grid showing Current Selected Song Detail
    st.markdown("### 🎶 Currently Selected Song")
    col_a, col_b = st.columns([1, 4])
    
    # Fetch artwork metadata
    meta = spotify_api.get_track_metadata(
        current_song.get("track_id", ""), 
        current_song["track_name"], 
        current_song["artist_name"]
    )
    
    with col_a:
        st.image(meta["artwork_url"], width=150)
    with col_b:
        st.markdown(f"## **{current_song['track_name']}**")
        st.markdown(f"#### **Artist**: {current_song['artist_name']}")
        st.markdown(f"**Album**: {current_song.get('album_name', 'N/A')}  |  **Year**: {current_song.get('year', 'N/A')}  |  **Language**: {current_song.get('language', 'N/A')}")
        
        # Action Buttons
        sub_c1, sub_c2 = st.columns([1, 3])
        with sub_c1:
            st.html(f'<a href="{meta["track_url"]}" target="_blank" class="spotify-btn">💚 Spotify Link</a>')
        
        with sub_c2:
            fav_id = f"{current_song['track_name']} - {current_song['artist_name']}"
            if fav_id not in st.session_state["favorites"]:
                if st.button("⭐ Add to Favorites", key="fav_btn"):
                    st.session_state["favorites"].append(fav_id)
                    st.success("Added to favorites!")
                    st.rerun()
            else:
                if st.button("💔 Remove from Favorites", key="fav_btn"):
                    st.session_state["favorites"].remove(fav_id)
                    st.warning("Removed from favorites!")
                    st.rerun()

    # Track Recommendation History
    history_entry = f"{current_song['track_name']} - {current_song['artist_name']}"
    if history_entry not in st.session_state["history"]:
        st.session_state["history"].append(history_entry)

    # Recommendations Section
    st.markdown("---")
    st.markdown("### 💡 Recommended for You")
    
    num_recs = st.slider("Number of recommendations:", min_value=3, max_value=12, value=config.DEFAULT_NUM_RECOMMENDATIONS)
    
    with st.spinner("Analyzing audio features & computing similarities..."):
        try:
            recommendations = recommender.recommend(selected_index, num_recommendations=num_recs)
        except Exception as e:
            st.error(f"Error fetching recommendations: {e}")
            recommendations = []

    if recommendations:
        # Create a grid layout (columns) for recommendation cards
        cols = st.columns(3)
        for i, rec in enumerate(recommendations):
            col_idx = i % 3
            with cols[col_idx]:
                # Fetch artwork and spotify link
                rec_meta = spotify_api.get_track_metadata(
                    rec.get("track_id", ""), 
                    rec["track_name"], 
                    rec["artist_name"]
                )
                
                # Render beautiful custom card
                st.html(
                    f"""
                    <div class="song-card">
                        <div style="text-align: center; margin-bottom: 8px;">
                            <img src="{rec_meta['artwork_url']}" style="width: 100%; max-height: 180px; object-fit: cover; border-radius: 8px;" />
                        </div>
                        <div>
                            <div class="song-title" title="{rec['track_name']}">{rec['track_name']}</div>
                            <div class="song-artist" title="{rec['artist_name']}">{rec['artist_name']}</div>
                            <div class="song-meta">💿 Album: {rec.get('album_name', 'N/A')}</div>
                            <div class="song-meta">📅 Year: {rec.get('year', 'N/A')} | Pop: {rec.get('popularity', 'N/A')}</div>
                        </div>
                        <div class="song-explanation">
                            💡 {rec['explanation']}
                        </div>
                        <div style="margin-top: 8px;">
                            <a href="{rec_meta['track_url']}" target="_blank" class="spotify-btn" style="width:100%;">Listen on Spotify</a>
                        </div>
                    </div>
                    """
                )
    else:
        st.info("No recommendations found.")

# Secondary features: Favorites and History
st.markdown("---")
tab1, tab2 = st.tabs(["⭐ Your Favorites", "🕒 Search History"])

with tab1:
    if st.session_state["favorites"]:
        for fav in st.session_state["favorites"]:
            st.markdown(f"- 🎵 **{fav}**")
    else:
        st.info("No favorite songs added yet.")

with tab2:
    if st.session_state["history"]:
        # Show last 10 entries in history
        for hist in reversed(st.session_state["history"][-10:]):
            st.markdown(f"- 🕒 **{hist}**")
    else:
        st.info("History is empty.")
