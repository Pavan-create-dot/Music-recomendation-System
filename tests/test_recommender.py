import os
import sys
import unittest
import pandas as pd
import numpy as np

# Adjust path to import config and src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.data_loader import DataLoader
from src.recommender import MusicRecommender

class TestMusicRecommenderSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a mock dataset to test components in isolation
        cls.mock_data = pd.DataFrame({
            'track_id': ['1', '2', '3', '4', '5'],
            'track_name': ['Song One', 'Song Two', 'Song Three', 'Song Four', 'Song Five'],
            'artist_name': ['Artist A', 'Artist A', 'Artist B', 'Artist C', 'Artist C'],
            'album_name': ['Album X', 'Album X', 'Album Y', 'Album Z', 'Album Z'],
            'year': [2020, 2020, 2021, 2022, 2022],
            'popularity': [80, 75, 60, 90, 85],
            'language': ['English', 'English', 'Tamil', 'Spanish', 'Spanish'],
            'danceability': [0.8, 0.75, 0.5, 0.9, 0.88],
            'energy': [0.7, 0.68, 0.4, 0.8, 0.85],
            'valence': [0.6, 0.58, 0.3, 0.9, 0.85],
            'tempo': [120.0, 118.0, 95.0, 128.0, 130.0],
            'track_url': ['url1', 'url2', 'url3', 'url4', 'url5']
        })
        
        # Build features soup for the mock data
        cls.mock_data['soup'] = cls.mock_data.apply(
            lambda row: f"{row['track_name']} {row['artist_name']} {row['album_name']} {row['year']} {row['language']}",
            axis=1
        )
        
        # Setup loader sub-class to avoid reading disk
        class MockDataLoader(DataLoader):
            def load_data(self):
                return cls.mock_data.copy()
                
        cls.recommender = MusicRecommender(data_loader=MockDataLoader())
        cls.recommender.initialize(force_recompute=True)

    def test_recommender_initialization(self):
        self.assertIsNotNone(self.recommender.df)
        self.assertIsNotNone(self.recommender.cosine_sim)
        self.assertEqual(len(self.recommender.df), 5)
        self.assertEqual(self.recommender.cosine_sim.shape, (5, 5))

    def test_track_index_retrieval(self):
        idx = self.recommender.get_track_index("Song One", "Artist A")
        self.assertEqual(idx, 0)
        
        invalid_idx = self.recommender.get_track_index("Nonexistent", "NoArtist")
        self.assertIsNone(invalid_idx)

    def test_recommendation_results(self):
        # Recommend for 'Song One' (Index 0, artist 'Artist A')
        recommendations = self.recommender.recommend(track_index=0, num_recommendations=2)
        
        self.assertEqual(len(recommendations), 2)
        # Song Two should be highly recommended as it's by the same artist and same album
        first_rec = recommendations[0]
        self.assertEqual(first_rec['track_name'], 'Song Two')
        self.assertEqual(first_rec['artist_name'], 'Artist A')
        
        # Check explanation tags are created
        self.assertTrue('explanation' in first_rec)
        self.assertTrue('by the same artist' in first_rec['explanation'] or 'from the same album' in first_rec['explanation'])

if __name__ == '__main__':
    unittest.main()
