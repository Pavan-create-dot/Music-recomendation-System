import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Dict, Any, Optional
from config import config
from src.utils import setup_logger

logger = setup_logger("spotify_api")

class SpotifyAPI:
    """
    Wrapper for Spotify Web API to fetch album covers and tracks links.
    Features graceful degradation if credentials are missing or API rate limits are hit.
    """
    def __init__(self):
        self.sp = None
        self.enabled = False
        
        client_id = config.SPOTIPY_CLIENT_ID
        client_secret = config.SPOTIPY_CLIENT_SECRET

        if not client_id or not client_secret:
            logger.warning("Spotify Client ID or Secret not configured. Running in offline/placeholder mode.")
            return

        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id, 
                client_secret=client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            # Try a simple call to verify it works (or just set enabled)
            self.enabled = True
            logger.info("Spotify API initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Spotipy: {e}. Running in fallback mode.")
            self.sp = None
            self.enabled = False

    def get_track_metadata(self, track_id: str, track_name: str, artist_name: str) -> Dict[str, Any]:
        """
        Fetches track metadata (album cover image URL, play link) from Spotify.
        Falls back to beautiful placeholder images if Spotify is unavailable.
        """
        fallback_data = {
            "artwork_url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
            "track_url": f"https://open.spotify.com/search/{track_name.replace(' ', '%20')}",
            "artist_info": "Artist information unavailable."
        }

        if not self.enabled or not self.sp:
            return fallback_data

        try:
            # If we have a track_id, try to fetch details directly
            if track_id and len(str(track_id).strip()) > 4:
                track = self.sp.track(track_id)
                images = track['album'].get('images', [])
                artwork_url = images[0]['url'] if images else fallback_data['artwork_url']
                track_url = track['external_urls'].get('spotify', fallback_data['track_url'])
                return {
                    "artwork_url": artwork_url,
                    "track_url": track_url
                }
            
            # If direct ID fetch failed, search using track name and artist
            query = f"track:{track_name} artist:{artist_name}"
            results = self.sp.search(q=query, type='track', limit=1)
            
            if results and results.get('tracks') and results['tracks'].get('items'):
                track = results['tracks']['items'][0]
                images = track['album'].get('images', [])
                artwork_url = images[0]['url'] if images else fallback_data['artwork_url']
                track_url = track['external_urls'].get('spotify', fallback_data['track_url'])
                return {
                    "artwork_url": artwork_url,
                    "track_url": track_url
                }
            
        except Exception as e:
            logger.warning(f"Error fetching Spotify metadata for {track_name} by {artist_name}: {e}")
            
        return fallback_data
