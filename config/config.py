import os
import logging
from typing import Dict, Any

# Project Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Dataset details
DATASET_PATH = os.path.join(BASE_DIR, "spotify_tracks.csv")
# Fallback target in the data folder if we want to organize it
ORGANIZED_DATASET_PATH = os.path.join(DATA_DIR, "spotify_tracks.csv")

# Recommendations Config
DEFAULT_NUM_RECOMMENDATIONS = 6
SAMPLE_SIZE = 5000  # Keeping it responsive and within memory limits

# Model cache paths
SIMILARITY_MATRIX_CACHE = os.path.join(MODEL_CACHE_DIR, "cosine_sim_matrix.pkl")
TFIDF_VECTORIZER_CACHE = os.path.join(MODEL_CACHE_DIR, "tfidf_vectorizer.pkl")
PROCESSED_DATA_CACHE = os.path.join(MODEL_CACHE_DIR, "processed_df.pkl")

# Logging Configuration
LOG_FILE = os.path.join(BASE_DIR, "music_recommender.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

# Spotify API credentials fallback
# If client ID and secret are set via env variables, we will use them.
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "33a98332e9c743cb86e1fb85a6669047")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "f846c1469df54735bdb7a0a80ec89f67")
