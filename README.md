# VibeSync - Content-Based Music Recommendation System

VibeSync is a clean, production-grade, and interview-ready **Content-Based Filtering Music Recommendation System** built using Python, Scikit-Learn, and Streamlit. The system recommends music based on metadata attributes (song name, artist, album, language, release year) and audio profile signatures.

---

## 📁 Modular Directory Structure

The project has been refactored into a highly modular layout following software engineering best practices:

```text
├── config/
│   └── config.py          # Centralized configuration settings and file path constants
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # CSV validation, cleaning, and features processing pipeline
│   ├── recommender.py     # Content-Based TF-IDF and Cosine Similarity engine (with model caching & explainability)
│   ├── spotify_api.py     # Spotify Web API wrapper for fetching album covers & music links
│   └── utils.py           # Log setup and project-level utility functions
├── tests/
│   └── test_recommender.py# Automation unit tests for correctness validation
├── app.py                 # Streamlit UI presentation layer
├── requirements.txt       # Dependencies manifest
└── music_recommender.log  # Output log file
```

---

## 🛠️ Features

1. **Robust Preprocessing Pipeline**: Cleans duplicate tracks (keeping highest popularity), standardizes casing, handles missing values, and processes text features into a "metadata soup".
2. **Cached Similarity Engine**: Serializes the TF-IDF matrix and Cosine Similarity model on the first load, reducing startup times to near instantaneous on subsequent launches.
3. **Explainable Recommendations**: Explains *why* each song was recommended (e.g. "by the same artist", "from the same album", "similar tempo", etc.).
4. **Spotify API Integration**: Seamlessly fetches track cover art and play links. Falls back to Unsplash placeholders and query search URLs if credentials are missing or API rate limits are hit.
5. **Modern Streamlit Dashboard**: Custom-styled dark mode with glassmorphic cards, responsive columns, sidebar filters, favorites checklist, and search history session tracking.

---

## 🚀 Installation & Running Locally

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Music-recomendation-System-main
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Spotify Credentials (Optional)**:
   The application works perfectly out of the box. To enable live Spotify album art, export your Spotify developer credentials to your environment:
   ```bash
   # Windows (CMD)
   set SPOTIPY_CLIENT_ID=your_client_id
   set SPOTIPY_CLIENT_SECRET=your_client_secret

   # Windows (PowerShell)
   $env:SPOTIPY_CLIENT_ID="your_client_id"
   $env:SPOTIPY_CLIENT_SECRET="your_client_secret"
   ```

4. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Running Unit Tests

Verify recommendation mechanics and model loading constraints with standard unit tests:
```bash
python -m unittest tests/test_recommender.py
```

---

## 🎓 Interview Questions & Preparation

### 1. Data Loader (`src/data_loader.py`)
- **Why it exists**: Isolates the data load, schema validation, and text soup generation from the presentation layer.
- **Complexity**: Time Complexity $O(N)$ for parsing the CSV and constructing text soup, where $N$ is the number of rows. Space Complexity $O(N)$ to hold the dataframe.
- **Common Questions**:
  - *How do you handle dataset updates or drift?* We implement an incremental preprocessing pipeline or schedule batch jobs to re-cache vectors.
  - *Why drop duplicate tracks by keeping the highest popularity?* To ensure recommenders select clean tracks with the best quality metadata.

### 2. Recommendation Engine (`src/recommender.py`)
- **Why it exists**: Builds and queries the Content-Based Filtering vector space model using TF-IDF Vectorization and Cosine Similarity.
- **Complexity**:
  - *Fitting*: Time Complexity $O(N \cdot D)$ to fit TF-IDF where $D$ is average vocabulary size, and $O(N^2 \cdot M)$ to compute Cosine Similarity between $N$ vectors of length $M$.
  - *Querying*: Time Complexity $O(N \log K)$ to fetch the top $K$ recommendations (using sorting/heaps).
  - *Space Complexity*: $O(N^2)$ to store the pairwise similarity matrix.
- **Common Questions**:
  - *Why TF-IDF instead of simple count vectorization?* TF-IDF down-weights terms that appear frequently across all tracks (like "mix" or "feat"), emphasizing unique tokens.
  - *How does the system handle cold-start issues?* Content-based filtering handles the item-cold-start issue well since it only relies on track metadata, not user interaction history.

### 3. Caching & Memory Optimizations
- **Why it exists**: Avoids recalculating the similarity matrix on every client interaction, reducing boot time from several seconds to milliseconds.
- **Common Questions**:
  - *What serialization protocol did you use?* Pickle, as it is native to Python and fast for numeric numpy arrays.
  - *How would you scale this for 1 million tracks?* Storing an $O(N^2)$ similarity matrix in memory becomes unfeasible. We would transition to Approximate Nearest Neighbors (ANN) libraries like Faiss or Spotify's Annoy, storing vector embeddings in a vector database like Pinecone, Milvus, or Qdrant.
