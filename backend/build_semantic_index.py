"""Build the optional embedding index without touching existing TF-IDF artifacts."""
import json
import os
import tempfile

import numpy as np
from sentence_transformers import SentenceTransformer
from app.main import (
    DF_PATH,
    EMBEDDINGS_META_PATH,
    EMBEDDINGS_PATH,
    _pickle_load_compatible,
    catalog_fingerprint,
)

MODEL = os.getenv("CINEMATCH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
frame = _pickle_load_compatible(DF_PATH)
columns = ["title", "overview", "genres", "keywords", "tagline", "original_title"]
documents = frame.reindex(columns=columns).fillna("").astype(str).agg(" ".join, axis=1).tolist()
encoder = SentenceTransformer(MODEL)
vectors = encoder.encode(documents, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
if vectors.ndim != 2 or vectors.shape[0] != len(frame) or not np.isfinite(vectors).all():
    raise RuntimeError("The encoder returned an invalid embedding matrix")
if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.02):
    raise RuntimeError("The generated vectors are not normalized")

temporary_path = None
try:
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(EMBEDDINGS_PATH), suffix=".npy", delete=False) as temp:
        temporary_path = temp.name
    np.save(temporary_path, vectors.astype(np.float32))
    os.replace(temporary_path, EMBEDDINGS_PATH)
finally:
    if temporary_path and os.path.exists(temporary_path):
        os.remove(temporary_path)

manifest = {
    "schema_version": 1,
    "model": MODEL,
    "rows": len(frame),
    "dimensions": int(vectors.shape[1]),
    "catalog_fingerprint": catalog_fingerprint(frame),
}
with open(EMBEDDINGS_META_PATH, "w", encoding="utf-8") as output:
    json.dump(manifest, output, indent=2)
print(f"Saved {len(vectors)} normalized vectors to {EMBEDDINGS_PATH}")
