import os
import pickle
import re
import json
import hashlib
import asyncio
import difflib
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "model", "artifacts")
DF_PATH = os.path.join(ARTIFACTS_DIR, "df.pkl")
INDICES_PATH = os.path.join(ARTIFACTS_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")
CATALOG_CSV_PATH = os.path.join(PROJECT_DIR, "model", "data", "movies_metadata.csv")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "semantic_embeddings.npy")
EMBEDDINGS_META_PATH = os.path.join(ARTIFACTS_DIR, "semantic_embeddings.meta.json")
EMBEDDING_MODEL = os.getenv("CINEMATCH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
semantic_embeddings: Optional[np.ndarray] = None
embedding_encoder: Any = None


class VibeFilters(BaseModel):
    genre: Optional[str] = Field(None, max_length=40)
    min_rating: Optional[float] = Field(None, ge=0, le=10)
    max_rating: Optional[float] = Field(None, ge=0, le=10)
    year_from: Optional[int] = Field(None, ge=1870, le=2100)
    year_to: Optional[int] = Field(None, ge=1870, le=2100)
    language: Optional[str] = Field(None, min_length=2, max_length=5, pattern=r"^[a-zA-Z]{2,5}$")
    min_runtime: Optional[int] = Field(None, ge=1, le=1000)
    max_runtime: Optional[int] = Field(None, ge=1, le=1000)


class VibeRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    filters: VibeFilters = Field(default_factory=VibeFilters)
    page: int = Field(1, ge=1, le=100)
    limit: int = Field(20, ge=1, le=50)
    personalize: bool = False
    liked_ids: List[int] = Field(default_factory=list, max_length=500)
    disliked_ids: List[int] = Field(default_factory=list, max_length=500)

    @field_validator("query")
    @classmethod
    def validate_query_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("query must contain at least 3 non-whitespace characters")
        return value

    @field_validator("liked_ids", "disliked_ids")
    @classmethod
    def validate_movie_ids(cls, value: List[int]) -> List[int]:
        if any(movie_id < 1 for movie_id in value):
            raise ValueError("movie IDs must be positive integers")
        return list(dict.fromkeys(value))


class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class VibeMovie(TMDBMovieCard):
    score: float
    explanation: str


class VibeResponse(BaseModel):
    query: str
    results: List[VibeMovie]
    page: int
    total: int
    ranking_mode: str


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


def _norm_title(t: str) -> str:
    value = str(t).strip().lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not TMDB_API_KEY:
        raise HTTPException(status_code=503, detail="TMDB integration is not configured")
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    last_error = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0)
            ) as client:
                response = await client.get(f"{TMDB_BASE}{path}", params=q)
            if response.status_code == 200:
                return response.json()
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise HTTPException(
                    status_code=502,
                    detail=f"TMDB error {response.status_code}: {response.text[:500]}",
                )
        except httpx.TimeoutException as e:
            last_error = e
            if attempt == 2:
                raise HTTPException(
                    status_code=504, detail="TMDB request timed out"
                ) from e
        except httpx.RequestError as e:
            last_error = e
            if attempt == 2:
                raise HTTPException(
                    status_code=502, detail=f"TMDB request error: {type(e).__name__}"
                ) from e
        await asyncio.sleep(0.75 * (attempt + 1))
    raise HTTPException(
        status_code=502,
        detail=f"TMDB request failed: {type(last_error).__name__ if last_error else 'unknown error'}",
    )


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    try:
        items = indices.items()
    except Exception as e:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like") from e
    for key, value in items:
        try:
            title_to_idx[_norm_title(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return title_to_idx


def build_df_title_map(frame: pd.DataFrame) -> Dict[str, int]:
    result: Dict[str, int] = {}
    if frame is None or "title" not in frame.columns:
        return result
    for position, title in enumerate(frame["title"].fillna("")):
        key = _norm_title(title)
        if key and key not in result:
            result[key] = position
    return result


def enrich_catalog_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore identifier and filter columns omitted from the saved recommendation dataframe."""
    wanted = ["title", "overview", "id", "release_date", "runtime", "original_language", "poster_path"]
    missing = [column for column in wanted[1:] if column not in frame.columns]
    if not missing or not os.path.isfile(CATALOG_CSV_PATH):
        return frame
    frame.columns = [str(column) for column in frame.columns]
    source = pd.read_csv(CATALOG_CSV_PATH, usecols=wanted, low_memory=False)
    def normalized(values: Any) -> pd.Series:
        return values.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    left_title, right_title = normalized(frame["title"]), normalized(source["title"])
    left_overview = normalized(frame.get("overview", pd.Series("", index=frame.index)))
    right_overview = normalized(source["overview"])
    left_key = left_title + "\x1f" + left_overview
    right_key = right_title + "\x1f" + right_overview
    left = pd.DataFrame({"__key": left_key.to_numpy(), "__occ": left_key.groupby(left_key).cumcount().to_numpy(), "__pos": np.arange(len(frame))})
    source["__key"] = right_key.to_numpy()
    source["__occ"] = right_key.groupby(right_key).cumcount().to_numpy()
    joined = left.merge(source.drop(columns=["title", "overview"]), on=["__key", "__occ"], how="left", sort=False).sort_values("__pos")
    for column in missing:
        frame[column] = joined[column].to_numpy()
    return frame


def catalog_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for title, overview in zip(frame["title"].fillna(""), frame["overview"].fillna("")):
        digest.update(str(title).encode("utf-8", errors="replace"))
        digest.update(b"\x1f")
        digest.update(str(overview).encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX, df, tfidf_matrix
    if TITLE_TO_IDX is None or df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    key = _norm_title(title)
    candidates = [key]
    key_without_year = re.sub(r"\s+(19|20)\d{2}$", "", key).strip()
    if key_without_year != key:
        candidates.append(key_without_year)

    for candidate in candidates:
        if candidate in TITLE_TO_IDX:
            idx = int(TITLE_TO_IDX[candidate])
            if 0 <= idx < len(df) and idx < tfidf_matrix.shape[0]:
                return idx

    matches = difflib.get_close_matches(
        key, list(TITLE_TO_IDX.keys()), n=1, cutoff=0.86
    )
    if matches:
        idx = int(TITLE_TO_IDX[matches[0]])
        if 0 <= idx < len(df) and idx < tfidf_matrix.shape[0]:
            return idx

    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    raw_scores = tfidf_matrix @ qv.T
    scores = (
        raw_scores.toarray().ravel()
        if hasattr(raw_scores, "toarray")
        else np.asarray(raw_scores).ravel()
    )
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        i = int(i)
        if i == idx or i >= len(df):
            continue
        score = float(scores[i])
        if score <= 0:
            continue
        title_value = str(df.iloc[i]["title"])
        if not title_value or title_value.lower() == "nan":
            continue
        out.append((title_value, score))
        if len(out) >= top_n:
            break
    return out


def local_movie_card(title: str) -> Optional[TMDBMovieCard]:
    if df is None or "title" not in df.columns:
        return None
    try:
        idx = get_local_idx_by_title(title)
        row = df.iloc[idx]
        raw_id = row.get("id")
        if pd.isna(raw_id):
            return None
        return TMDBMovieCard(
            tmdb_id=int(raw_id),
            title=str(row.get("title") or title),
            poster_url=make_img_url(row.get("poster_path")),
            release_date=None
            if pd.isna(row.get("release_date"))
            else str(row.get("release_date")),
            vote_average=None
            if pd.isna(row.get("vote_average"))
            else float(row.get("vote_average")),
        )
    except Exception:
        return None


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    local_card = local_movie_card(title)
    if local_card is not None:
        return local_card
    try:
        movie = await tmdb_search_first(title)
        if not movie:
            return None
        return TMDBMovieCard(
            tmdb_id=int(movie["id"]),
            title=movie.get("title") or title,
            poster_url=make_img_url(movie.get("poster_path")),
            release_date=movie.get("release_date"),
            vote_average=movie.get("vote_average"),
        )
    except Exception:
        return None


async def attach_tfidf_cards(recs: List[Tuple[str, float]]) -> List[TFIDFRecItem]:
    cards = await asyncio.gather(
        *(attach_tmdb_card_by_title(title) for title, _ in recs),
        return_exceptions=True,
    )
    result: List[TFIDFRecItem] = []
    for (title, score), card in zip(recs, cards):
        if isinstance(card, Exception):
            card = None
        result.append(TFIDFRecItem(title=title, score=score, tmdb=card))
    return result


def _pickle_load_compatible(path: str) -> Any:
    """Read serialized pandas 3 string columns on pandas versions without pyarrow storage."""
    try:
        with open(path, "rb") as artifact:
            return pickle.load(artifact)
    except TypeError:
        string_dtype = getattr(pd, "StringDtype", None)
        if string_dtype is None:
            raise
        original_init = string_dtype.__init__

        def compatible_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            storage = kwargs.get("storage", args[0] if args else None)
            if storage == "pyarrow":
                storage = "python"
            if storage is None:
                original_init(instance)
            else:
                original_init(instance, storage)

        string_dtype.__init__ = compatible_init
        try:
            with open(path, "rb") as artifact:
                return pickle.load(artifact)
        finally:
            string_dtype.__init__ = original_init


@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX, semantic_embeddings, embedding_encoder
    required = [DF_PATH, TFIDF_MATRIX_PATH, TFIDF_PATH]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    try:
        df = _pickle_load_compatible(DF_PATH)
    except Exception as exc:
        raise RuntimeError("Movie dataframe artifact is incompatible; refusing to guess its row alignment") from exc
    try:
        indices_obj = _pickle_load_compatible(INDICES_PATH)
    except Exception:
        indices_obj = {}
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    df = enrich_catalog_metadata(df)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
    for key, value in build_df_title_map(df).items():
        TITLE_TO_IDX.setdefault(key, value)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")
    if len(df) != tfidf_matrix.shape[0]:
        raise RuntimeError(
            f"df rows ({len(df)}) and TF-IDF rows ({tfidf_matrix.shape[0]}) do not match"
        )
    if not {"id", "title", "overview", "genres"}.issubset(df.columns):
        raise RuntimeError("Movie catalog is missing required identifiers or searchable fields")
    df = df.reset_index(drop=True)
    if os.path.isfile(EMBEDDINGS_PATH) and os.path.isfile(EMBEDDINGS_META_PATH):
        try:
            with open(EMBEDDINGS_META_PATH, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if (manifest.get("schema_version") != 1 or manifest.get("model") != EMBEDDING_MODEL or manifest.get("rows") != len(df) or manifest.get("catalog_fingerprint") != catalog_fingerprint(df)):
                raise ValueError("Semantic index does not match the current catalog ordering")
            loaded = np.load(EMBEDDINGS_PATH, mmap_mode="r")
            if loaded.ndim != 2 or loaded.shape[0] != len(df) or not np.isfinite(loaded).all():
                raise ValueError("Incompatible semantic index")
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            dimension = encoder.get_sentence_embedding_dimension()
            norms = np.linalg.norm(loaded, axis=1)
            if loaded.shape[1] != dimension or loaded.shape[1] != manifest.get("dimensions") or not np.allclose(norms, 1, atol=0.02):
                raise ValueError("Semantic index dimension or normalization mismatch")
            semantic_embeddings = loaded
            embedding_encoder = encoder
        except (OSError, ValueError):
            semantic_embeddings = None
            embedding_encoder = None
        except Exception:
            semantic_embeddings = None
            embedding_encoder = None


@app.get("/health")
def health():
    return {"status": "ok"}


def _row_text(row: Any) -> str:
    fields = ["title", "overview", "genres", "keywords", "tagline", "original_title"]
    return " ".join(str(row.get(name, "")) for name in fields if row.get(name) is not None)


def _filter_mask(filters: VibeFilters) -> np.ndarray:
    if df is None:
        return np.zeros(0, dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    if filters.genre:
        if "genres" not in df.columns:
            raise HTTPException(status_code=422, detail="Genre filtering is unavailable in this dataset")
        mask &= df["genres"].fillna("").astype(str).str.contains(re.escape(filters.genre), case=False, regex=True).to_numpy()
    if filters.min_rating is not None:
        if "vote_average" not in df.columns:
            raise HTTPException(status_code=422, detail="Rating filtering is unavailable in this dataset")
        mask &= pd.to_numeric(df["vote_average"], errors="coerce").fillna(-1).to_numpy() >= filters.min_rating
    if filters.max_rating is not None:
        if "vote_average" not in df.columns:
            raise HTTPException(status_code=422, detail="Rating filtering is unavailable in this dataset")
        ratings = pd.to_numeric(df["vote_average"], errors="coerce").fillna(11).to_numpy()
        mask &= ratings <= filters.max_rating
    if filters.year_from is not None or filters.year_to is not None:
        if "release_date" not in df.columns:
            raise HTTPException(status_code=422, detail="Release year filtering is unavailable in this dataset")
        years = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce").to_numpy()
        if filters.year_from is not None: mask &= years >= filters.year_from
        if filters.year_to is not None: mask &= years <= filters.year_to
    if filters.language:
        if "original_language" not in df.columns:
            raise HTTPException(status_code=422, detail="Language filtering is unavailable in this dataset")
        mask &= df["original_language"].fillna("").astype(str).str.lower().to_numpy() == filters.language.lower()
    if filters.min_runtime is not None or filters.max_runtime is not None:
        if "runtime" not in df.columns:
            raise HTTPException(status_code=422, detail="Runtime filtering is unavailable in this dataset")
        runtime = pd.to_numeric(df["runtime"], errors="coerce").fillna(-1).to_numpy()
        if filters.min_runtime is not None: mask &= runtime >= filters.min_runtime
        if filters.max_runtime is not None: mask &= runtime <= filters.max_runtime
    return mask


def vibe_rank(request: VibeRequest) -> Tuple[np.ndarray, str]:
    if df is None:
        raise HTTPException(status_code=503, detail="Movie catalog is unavailable")
    if request.filters.year_from is not None and request.filters.year_to is not None and request.filters.year_from > request.filters.year_to:
        raise HTTPException(status_code=422, detail="year_from must be no later than year_to")
    if request.filters.min_runtime is not None and request.filters.max_runtime is not None and request.filters.min_runtime > request.filters.max_runtime:
        raise HTTPException(status_code=422, detail="min_runtime must be no greater than max_runtime")
    if request.filters.min_rating is not None and request.filters.max_rating is not None and request.filters.min_rating > request.filters.max_rating:
        raise HTTPException(status_code=422, detail="min_rating must be no greater than max_rating")
    semantic_available = semantic_embeddings is not None and embedding_encoder is not None
    tfidf_available = tfidf_obj is not None and tfidf_matrix is not None
    if not semantic_available and not tfidf_available:
        raise HTTPException(status_code=503, detail="No supported movie search index is available")
    try:
        if not semantic_available:
            raise RuntimeError("semantic index unavailable")
        if semantic_embeddings.shape[0] != len(df) or semantic_embeddings.shape[1] != embedding_encoder.get_sentence_embedding_dimension():
            raise RuntimeError("semantic index shape mismatch")
        query_vector = np.asarray(embedding_encoder.encode([request.query], normalize_embeddings=True)[0])
        if query_vector.shape[0] != semantic_embeddings.shape[1] or not np.isfinite(query_vector).all():
            raise RuntimeError("invalid semantic query vector")
        scores = np.asarray(semantic_embeddings @ query_vector).reshape(-1)
        mode = "sentence-transformer"
    except Exception:
        if not tfidf_available:
            raise HTTPException(status_code=503, detail="Semantic search is unavailable and TF-IDF fallback is missing")
        try:
            query_vector = tfidf_obj.transform([request.query])
            raw = tfidf_matrix @ query_vector.T
            scores = raw.toarray().ravel() if hasattr(raw, "toarray") else np.asarray(raw).ravel()
            mode = "tfidf-fallback"
        except Exception as exc:
            raise HTTPException(status_code=503, detail="TF-IDF search index is incompatible") from exc
    scores = np.clip(scores, 0, 1)
    if request.personalize and request.liked_ids and tfidf_matrix is not None:
        liked_rows = df.index[pd.to_numeric(df.get("id"), errors="coerce").isin(request.liked_ids)].to_numpy()
        liked_rows = liked_rows[liked_rows < len(scores)]
        if len(liked_rows):
            taste = np.asarray(tfidf_matrix[liked_rows].mean(axis=0))
            personal = tfidf_matrix @ taste.T
            personal = personal.toarray().ravel() if hasattr(personal, "toarray") else np.asarray(personal).ravel()
            personal = np.clip(personal, 0, 1)
            scores = .78 * scores + .22 * personal
    mask = _filter_mask(request.filters)
    valid_titles = df["title"].fillna("").astype(str).str.strip().ne("").to_numpy()
    valid_ids = pd.to_numeric(df.get("id"), errors="coerce")
    mask &= valid_titles & valid_ids.notna().to_numpy()
    mask &= ~valid_ids.duplicated(keep="first").to_numpy()
    if request.disliked_ids and "id" in df:
        mask &= ~pd.to_numeric(df["id"], errors="coerce").isin(request.disliked_ids).to_numpy()
    return np.where(mask, scores, -1), mode


@app.post("/discover/vibe", response_model=VibeResponse)
async def discover_vibe(request: VibeRequest):
    scores, mode = vibe_rank(request)
    start = (request.page - 1) * request.limit
    order = np.argsort(-scores, kind="stable")
    ranked = order[scores[order] > 0]
    total = len(ranked)
    results = []
    for idx in ranked[start : start + request.limit]:
        row = df.iloc[int(idx)]
        try:
            movie_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        title = str(row.get("title") or "Untitled")
        genre = str(row.get("genres") or "")
        explanation = f"Matched your description using catalog text{f'; catalog genres include {genre}' if genre else ''}."
        liked_resolved = request.personalize and bool(
            pd.to_numeric(df["id"], errors="coerce").isin(request.liked_ids).any()
        )
        if liked_resolved:
            explanation += " Ranking also considered movies you liked."
        results.append(VibeMovie(
            tmdb_id=movie_id, title=title,
            poster_url=make_img_url(row.get("poster_path")),
            release_date=None if pd.isna(row.get("release_date")) else str(row.get("release_date")),
            vote_average=None if pd.isna(row.get("vote_average")) else float(row.get("vote_average")),
            score=round(float(scores[idx]), 4), explanation=explanation,
        ))
    return VibeResponse(query=request.query, results=results, page=request.page, total=total, ranking_mode=mode)


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        elif category in {"popular", "top_rated", "upcoming", "now_playing"}:
            data = await tmdb_get(
                f"/movie/{category}", {"language": "en-US", "page": 1}
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid category")
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1), page: int = Query(1, ge=1, le=10)
):
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...), limit: int = Query(18, ge=1, le=50)
):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [card for card in cards if card.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1), top_n: int = Query(10, ge=1, le=50)
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": title, "score": score} for title, score in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    recs: List[Tuple[str, float]] = []
    for candidate in (details.title, query):
        try:
            recs = tfidf_recommend_titles(candidate, top_n=tfidf_top_n)
            if recs:
                break
        except HTTPException:
            continue
        except Exception:
            continue

    tfidf_items = await attach_tfidf_cards(recs)

    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [card for card in cards if card.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
