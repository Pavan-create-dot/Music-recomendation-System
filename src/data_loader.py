import os
import pandas as pd
from typing import Tuple, List
from config import config
from src.utils import setup_logger

logger = setup_logger("data_loader")

class DataLoader:
    """
    Handles data ingestion, metadata validation, cleaning, and preprocessing pipeline.
    """
    def __init__(self, filepath: str = config.DATASET_PATH):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset, handles missing values, removes duplicates, and performs basic validation.
        """
        logger.info(f"Checking for dataset at: {self.filepath}")
        if not os.path.exists(self.filepath):
            # Try fallback organized dataset path
            if os.path.exists(config.ORGANIZED_DATASET_PATH):
                self.filepath = config.ORGANIZED_DATASET_PATH
                logger.info(f"Using fallback organized dataset at: {self.filepath}")
            else:
                err_msg = f"Dataset not found at {self.filepath} or {config.ORGANIZED_DATASET_PATH}!"
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

        logger.info("Loading music tracks dataset...")
        try:
            # Load only necessary columns to optimize memory usage
            df = pd.read_csv(self.filepath)
        except Exception as e:
            logger.error(f"Failed to read CSV dataset: {e}")
            raise e

        initial_shape = df.shape
        logger.info(f"Dataset loaded successfully. Shape: {initial_shape}")

        # Basic validation: ensure critical metadata columns exist
        required_cols = ['track_id', 'track_name', 'artist_name', 'album_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            err_msg = f"Dataset is missing required columns: {missing_cols}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Handle missing values in key metadata columns
        df.dropna(subset=['track_name', 'artist_name', 'album_name'], inplace=True)
        after_dropna_shape = df.shape
        logger.info(f"Dropped {initial_shape[0] - after_dropna_shape[0]} rows due to missing critical track/artist/album name metadata.")

        # Standardize metadata casing and string format
        for col in ['track_name', 'artist_name', 'album_name']:
            df[col] = df[col].astype(str).str.strip()

        # Handle duplicates: Keep the one with highest popularity
        if 'popularity' in df.columns:
            df = df.sort_values(by='popularity', ascending=False)
        
        # Remove duplicate track entries (same track name and artist)
        df.drop_duplicates(subset=['track_name', 'artist_name'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        after_duplicates_shape = df.shape
        logger.info(f"Dropped {after_dropna_shape[0] - after_duplicates_shape[0]} duplicate tracks. Cleaned dataset size: {after_duplicates_shape}")

        # Limit to sample size to keep performance responsive and within constraints
        if len(df) > config.SAMPLE_SIZE:
            logger.info(f"Limiting dataset to first {config.SAMPLE_SIZE} tracks for optimized execution.")
            df = df.head(config.SAMPLE_SIZE).copy()

        # Build feature soup for recommendation engine
        df['soup'] = df.apply(self._create_feature_soup, axis=1)
        
        return df

    def _create_feature_soup(self, row: pd.Series) -> str:
        """
        Creates a 'feature soup' string for TF-IDF vectorization.
        Combines track metadata, artist, album, language and release year to build a rich feature profile.
        """
        track = str(row.get('track_name', ''))
        artist = str(row.get('artist_name', ''))
        album = str(row.get('album_name', ''))
        year = str(row.get('year', ''))
        language = str(row.get('language', ''))
        
        # We can also add other metadata or features if present
        soup_components = [track, artist, album, year, language]
        return " ".join([comp for comp in soup_components if comp])
