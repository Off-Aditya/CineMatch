const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, params = {}) {
  const url = new URL(`${API_BASE}${path}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });

  const response = await fetch(url);
  let data = null;

  try {
    data = await response.json();
  } catch {}

  if (!response.ok) {
    throw new Error(data?.detail || `Request failed (${response.status})`);
  }

  return data;
}

async function post(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.detail || `Request failed (${response.status})`);
  return data;
}

export const api = {
  health: () => request("/health"),
  home: (category = "popular", limit = 24) =>
    request("/home", { category, limit }),
  search: (query, page = 1) =>
    request("/tmdb/search", { query, page }),
  details: (tmdbId) =>
    request(`/movie/id/${tmdbId}`),
  bundle: (query, tfidfTopN = 12, genreLimit = 12) =>
    request("/movie/search", {
      query,
      tfidf_top_n: tfidfTopN,
      genre_limit: genreLimit,
    }),
  tfidf: (title, topN = 12) =>
    request("/recommend/tfidf", {
      title,
      top_n: topN,
    }),
  genreRecommendations: (tmdbId, limit = 18) =>
    request("/recommend/genre", {
      tmdb_id: tmdbId,
      limit,
    }),
  vibe: (payload) => post("/discover/vibe", payload),
};

export { API_BASE };
