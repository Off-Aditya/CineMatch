const STORAGE_KEY = "cinematch.profile.v1";

function newId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function freshProfile(profileId = newId()) {
  return { schemaVersion: 2, profileId, watchlist: [], liked: [], disliked: [], moviesById: {} };
}

function normalizeIds(values, moviesById) {
  if (!Array.isArray(values)) return [];
  return [...new Set(values.map((value) => {
    if (value && typeof value === "object") {
      const id = Number(value.tmdb_id ?? value.id);
      if (Number.isInteger(id) && id > 0) moviesById[id] = {
        tmdb_id: id, title: String(value.title || "Untitled"), poster_url: value.poster_url || null,
        release_date: value.release_date || "", vote_average: Number.isFinite(Number(value.vote_average)) ? Number(value.vote_average) : null,
      };
      return id;
    }
    const id = Number(value);
    return Number.isInteger(id) && id > 0 ? id : null;
  }).filter(Boolean))];
}

export function readProfile() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if ([1, 2].includes(value?.schemaVersion) && typeof value.profileId === "string") {
      const moviesById = {};
      if (value.moviesById && typeof value.moviesById === "object" && !Array.isArray(value.moviesById)) {
        for (const [key, movie] of Object.entries(value.moviesById)) {
          const id = Number(key);
          if (!Number.isInteger(id) || id < 1 || !movie || typeof movie !== "object") continue;
          moviesById[id] = { tmdb_id: id, title: String(movie.title || "Untitled"), poster_url: typeof movie.poster_url === "string" ? movie.poster_url : null, release_date: typeof movie.release_date === "string" ? movie.release_date : "", vote_average: Number.isFinite(Number(movie.vote_average)) ? Number(movie.vote_average) : null };
        }
      }
      const profile = freshProfile(value.profileId);
      profile.moviesById = moviesById;
      profile.watchlist = normalizeIds(value.watchlist, moviesById);
      profile.liked = normalizeIds(value.liked, moviesById);
      profile.disliked = normalizeIds(value.disliked, moviesById);
      profile.liked = profile.liked.filter((id) => !profile.disliked.includes(id));
      if (value.schemaVersion !== 2 || JSON.stringify(profile) !== JSON.stringify(value)) saveProfile(profile);
      return profile;
    }
  } catch {}
  const profile = freshProfile();
  saveProfile(profile);
  return profile;
}

export function saveProfile(profile) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(profile)); } catch {}
}

export function toggleMovieFeedback(profile, movie, field) {
  const id = Number(movie?.tmdb_id);
  if (!Number.isInteger(id) || id < 1 || !["liked", "disliked", "watchlist"].includes(field)) return profile;
  const next = { ...profile, moviesById: { ...profile.moviesById } };
  next.moviesById[id] = {
    tmdb_id: id, title: String(movie.title || "Untitled"), poster_url: movie.poster_url || null,
    release_date: movie.release_date || "", vote_average: movie.vote_average ?? null,
  };
  const exists = next[field].includes(id);
  next[field] = exists ? next[field].filter((movieId) => movieId !== id) : [...next[field], id];
  if (field === "liked" && !exists) next.disliked = next.disliked.filter((movieId) => movieId !== id);
  if (field === "disliked" && !exists) next.liked = next.liked.filter((movieId) => movieId !== id);
  if (![...next.liked, ...next.disliked, ...next.watchlist].includes(id)) delete next.moviesById[id];
  return next;
}

export function resetProfile() {
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
  return freshProfile();
}
