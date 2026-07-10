import os
import pickle
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from src.utils import setup_logger
from src.data_loader import DataLoader

logger = setup_logger("recommender")

class MusicRecommender:
    """
    Content-Based Filtering Recommendation Engine with support for:
    - Preprocessing & Feature Extraction (TF-IDF)
    - Cosine Similarity matrix calculation
    - Model caching (persistence and loading to reduce startup overhead)
    - Explainability (providing natural language reasons for recommendations)
    """
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
        self.df: Optional[pd.DataFrame] = None
        self.tfidf_matrix: Any = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.cosine_sim: Any = None

    def initialize(self, force_recompute: bool = False) -> None:
        """
        Initializes the recommender by loading data and computing or retrieving the cached similarity matrix.
        """
        if not force_recompute and self._load_cached_models():
            logger.info("Recommendation model successfully loaded from cache.")
            return

        logger.info("Cache missing or recompute forced. Building recommendation model from scratch...")
        self.df = self.data_loader.load_data()
        
        logger.info("Fitting TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['soup'])
        
        logger.info("Computing Cosine Similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        self._save_models_to_cache()
        logger.info("Recommendation model built and cached successfully.")

    def _load_cached_models(self) -> bool:
        """
        Loads models from cache files. Returns True if all files exist and load successfully.
        """
        cache_files = [
            config.SIMILARITY_MATRIX_CACHE,
            config.TFIDF_VECTORIZER_CACHE,
            config.PROCESSED_DATA_CACHE
        ]
        if not all(os.path.exists(f) for f in cache_files):
            return False

        try:
            with open(config.SIMILARITY_MATRIX_CACHE, "rb") as f:
                self.cosine_sim = pickle.load(f)
            with open(config.TFIDF_VECTORIZER_CACHE, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(config.PROCESSED_DATA_CACHE, "rb") as f:
                self.df = pickle.load(f)
            
            # Double check shapes match
            if self.df is not None and self.cosine_sim is not None:
                if len(self.df) == self.cosine_sim.shape[0]:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Will rebuild model.")
            return False

    def _save_models_to_cache(self) -> None:
        """
        Saves computed similarity, vectorizer, and processed dataframe to cache files.
        """
        try:
            with open(config.SIMILARITY_MATRIX_CACHE, "wb") as f:
                pickle.dump(self.cosine_sim, f)
            with open(config.TFIDF_VECTORIZER_CACHE, "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(config.PROCESSED_DATA_CACHE, "wb") as f:
                pickle.dump(self.df, f)
            logger.info("Successfully serialized models and dataframe to cache.")
        except Exception as e:
            logger.error(f"Failed to save models to cache: {e}")

    def get_track_index(self, track_name: str, artist_name: str) -> Optional[int]:
        """
        Gets the index of a track given the name and artist.
        """
        if self.df is None:
            return None
        matches = self.df[(self.df['track_name'].str.lower() == track_name.lower()) & 
                          (self.df['artist_name'].str.lower() == artist_name.lower())]
        if not matches.empty:
            return int(matches.index[0])
        return None

    def recommend(self, track_index: int, num_recommendations: int = config.DEFAULT_NUM_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Generates track recommendations with reasoning explanation based on cosine similarity and metadata overlap.
        """
        if self.df is None or self.cosine_sim is None:
            raise ValueError("Recommender is not initialized. Call initialize() first.")
        
        if track_index < 0 or track_index >= len(self.df):
            raise IndexError("Track index is out of bounds.")

        # Get pair-wise similarity scores
        sim_scores = list(enumerate(self.cosine_sim[track_index]))
        # Sort based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the track itself (index 0 in sorted list will be the track itself if it's 1.0 similarity)
        results: List[Dict[str, Any]] = []
        target_track = self.df.iloc[track_index]
        
        count = 0
        for idx, score in sim_scores:
            if idx == track_index:
                continue
            if count >= num_recommendations:
                break
                
            rec_track = self.df.iloc[idx]
            explanation = self._generate_explanation(target_track, rec_track)
            
            track_info = rec_track.to_dict()
            track_info['similarity_score'] = float(score)
            track_info['explanation'] = explanation
            results.append(track_info)
            count += 1

        return results

    def _generate_explanation(self, target: pd.Series, candidate: pd.Series) -> str:
        """
        Compares metadata values of the target and candidate track to formulate an explanation.
        """
        reasons = []
        
        # Check artist match
        if str(target.get('artist_name')).lower() == str(candidate.get('artist_name')).lower():
            reasons.append(f"by the same artist ({target.get('artist_name')})")
            
        # Check language match
        if 'language' in target and 'language' in candidate:
            if target['language'] == candidate['language'] and pd.notna(target['language']):
                # Don't make it the only reason unless it is unique, but it adds support
                pass

        # Check album match
        if str(target.get('album_name')).lower() == str(candidate.get('album_name')).lower() and pd.notna(target.get('album_name')):
            reasons.append("from the same album")

        # Compare audio features if present in the dataset
        audio_matches = []
        for feature in ['danceability', 'energy', 'valence', 'tempo']:
            if feature in target and feature in candidate:
                val_t = target[feature]
                val_c = candidate[feature]
                # If difference is within 12%
                if pd.notna(val_t) and pd.notna(val_c):
                    if feature == 'tempo':
                        diff = abs(val_t - val_c) / val_t
                        if diff < 0.10:
                            audio_matches.append("similar rhythm/tempo")
                    else:
                        diff = abs(val_t - val_c)
                        if diff < 0.12:
                            audio_matches.append(f"similar {feature}")

        if audio_matches:
            # Pick the top audio features
            reasons.append(", ".join(audio_matches[:2]))

        if not reasons:
            reasons.append("shared musical characteristics and metadata attributes")

        # Capitalize and format
        explanation = "Recommended because of " + " & ".join(reasons)
        return explanation
