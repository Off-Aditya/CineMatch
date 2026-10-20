# CineMatch

CineMatch is a React/Vite and FastAPI movie-discovery app. Existing TMDB browsing, search, movie details, genre suggestions, and title-to-title TF-IDF recommendations remain in place. The added Discover view accepts natural-language prompts, and a browser-local profile can adjust its rankings.

## Implemented features

- `POST /discover/vibe` accepts a 3–500 character query, optional supported filters, pagination, and optional like/dislike TMDB IDs. Responses contain movie cards, a `[0, 1]` ranking score, an evidence-based explanation, total result count, and `ranking_mode` (`sentence-transformer` or `tfidf-fallback`). Scores are ranking signals, not probabilities.
- Filters use local catalog fields: genre, rating range, release year range, original language, and runtime range. The loader restores TMDB IDs and these metadata columns by matching each saved dataframe row on title plus overview against `model/data/movies_metadata.csv`; this disambiguates duplicate titles. It preserves dataframe and TF-IDF row order. Invalid ranges or unavailable columns return HTTP 422.
- Semantic discovery is optional. With a matching sentence-transformer index and locally available model, queries are ranked by cosine similarity. Otherwise, the existing fitted TF-IDF vectorizer and matrix provide a lexical query-to-catalog fallback. The fallback does not provide full semantic understanding.
- When personalization is enabled and liked IDs resolve locally, the score is `0.78 × query_relevance + 0.22 × mean_TFIDF_similarity_to_liked_movies`. If none resolve, ranking is query-only. Disliked IDs are excluded. Unknown IDs are ignored. The home page also shows recommendations based on local likes.
- The profile uses `localStorage` key `cinematch.profile.v1` and schema version 2. It stores an anonymous ID, liked/disliked/watchlist TMDB ID arrays, and a small local movie-card cache so saved lists can render. Like and dislike are mutually exclusive. The Watchlist view shows counts and has a confirmation step before reset. Reset deletes only CineMatch’s storage key.

The current profile does not sync between devices. The profile ID and card cache stay in the browser. Only the liked/disliked IDs needed for ranking are sent with personalized vibe requests. With the Discover personalization toggle off, those arrays are omitted. Clearing browser storage deletes the profile.

Onboarding, watched-history tracking, explicit ratings, genre-preference learning, accounts, and server-side profile persistence are not implemented. The anonymous profile is local preference data, not an account or a trained model.

## Setup

### Backend

From `backend`:

```powershell
python -m pip install -r requirements.txt
```

Set `TMDB_API_KEY` in `backend/.env` for TMDB-backed routes. Set `CINEMATCH_EMBEDDING_MODEL` to choose the semantic model; the default is `sentence-transformers/all-MiniLM-L6-v2`. Start the API from `backend`:

```powershell
uvicorn app.main:app --reload
```

The app requires the existing `model/artifacts/df.pkl`, `tfidf_matrix.pkl`, and `tfidf.pkl`, plus `model/data/movies_metadata.csv` to restore IDs and filter metadata omitted from the saved dataframe. `indices.pkl` is used when readable; title indices can be rebuilt from the dataframe. The dataframe artifact must match the saved TF-IDF matrix row count. A pandas string-dtype compatibility loader maps serialized `pyarrow` strings to Python strings without reordering rows. Never replace the existing model artifacts as part of semantic-index setup.

The repository’s pinned scikit-learn version differs from the version recorded in the serialized TF-IDF estimator. It loads in the checked environment with an `InconsistentVersionWarning`; the existing and new TF-IDF paths were exercised there, but use a matching sklearn environment for reproducible model behavior.

### Frontend

From `frontend`:

```powershell
npm install
npm run dev
```

Set `VITE_API_BASE_URL` if FastAPI is not at `http://127.0.0.1:8000`.

## Optional semantic index

The default install does not download an embedding model. To enable semantic discovery, from `backend`:

```powershell
python -m pip install -r requirements-semantic.txt
python build_semantic_index.py
```

The builder downloads the configured model if needed, reads the existing dataframe, and writes `model/artifacts/semantic_embeddings.npy` plus a catalog/model manifest at `semantic_embeddings.meta.json`. The manifest checks model name, dimensions, row count, and a fingerprint of ordered titles and overviews. At backend startup the vectors are checked for compatibility, finite values, and normalization. Missing, corrupt, mismatched, or unavailable semantic assets fall back to TF-IDF. These files are separate from the existing TF-IDF artifacts.

## Vibe API

Example request:

```json
{
  "query": "A mysterious and emotional science-fiction story",
  "filters": {
    "genre": "Science Fiction",
    "min_rating": 6,
    "max_rating": 10,
    "year_from": 1990,
    "year_to": 2025,
    "language": "en",
    "min_runtime": 80,
    "max_runtime": 180
  },
  "page": 1,
  "limit": 20,
  "personalize": true,
  "liked_ids": [11, 22],
  "disliked_ids": [33]
}
```

Queries are trimmed and must contain 3–500 non-whitespace characters. IDs must be positive integers; each list is limited to 500. Page is 1–100 and limit is 1–50. Invalid filter ordering and unsupported catalog columns return 422. The endpoint does not need TMDB credentials because results come from the local catalog; TMDB is still required for browsing/search/details.

## Checks

From `backend`:

```powershell
python -m unittest discover -s tests -v
```

From `frontend`:

```powershell
npm test
npm run build
```

Backend unit tests use small in-memory fixtures and do not call TMDB or download a model. Frontend Node tests cover profile storage, malformed-data repair, migration, and reset behavior. They do not render-test the React UI. No relevance-labeled evaluation data is supplied, so no offline ranking-quality claim is made.
