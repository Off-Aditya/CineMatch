import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { readProfile, saveProfile, resetProfile, toggleMovieFeedback } from "./profile";

const categories = [
  ["trending", "Trending"],
  ["popular", "Popular"],
  ["top_rated", "Top Rated"],
  ["now_playing", "Now Playing"],
  ["upcoming", "Upcoming"],
];

const yearOf = (date) => (date ? String(date).slice(0, 4) : "");
const scoreOf = (value) =>
  Number.isFinite(Number(value)) ? Number(value).toFixed(1) : "—";

function MovieCard({ movie, onOpen, score }) {
  const title = movie?.title || "Untitled";
  const canOpen = Boolean(movie?.tmdb_id);

  return (
    <article
      className={`movie-card ${canOpen ? "clickable" : ""}`}
      onClick={() => canOpen && onOpen(movie.tmdb_id)}
      role={canOpen ? "button" : undefined}
      tabIndex={canOpen ? 0 : undefined}
      onKeyDown={(event) => {
        if (canOpen && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          onOpen(movie.tmdb_id);
        }
      }}
    >
      <div className="poster-wrap">
        {movie?.poster_url ? (
          <img src={movie.poster_url} alt={title} loading="lazy" />
        ) : (
          <div className="poster-empty">🎬</div>
        )}
        <div className="poster-gradient" />
        {movie?.vote_average != null && (
          <span className="rating-pill">★ {scoreOf(movie.vote_average)}</span>
        )}
        {score !== undefined && (
          <span className="match-pill">Score {Number(score).toFixed(2)}</span>
        )}
      </div>
      <div className="card-info">
        <h3>{title}</h3>
        <div className="card-meta">
          <span>{yearOf(movie.release_date) || "—"}</span>
          {score !== undefined && <span>Relevance score</span>}
        </div>
      </div>
    </article>
  );
}

function MovieFeedback({ movie, profile, onFeedback }) {
  const id = movie?.tmdb_id;
  if (!id) return null;
  return (
    <div className="card-actions">
      <button aria-pressed={profile.liked.includes(id)} onClick={() => onFeedback(movie, "liked")}>{profile.liked.includes(id) ? "Liked ✓" : "Like"}</button>
      <button aria-pressed={profile.disliked.includes(id)} onClick={() => onFeedback(movie, "disliked")}>{profile.disliked.includes(id) ? "Hidden ✓" : "Not for me"}</button>
      <button aria-pressed={profile.watchlist.includes(id)} onClick={() => onFeedback(movie, "watchlist")}>{profile.watchlist.includes(id) ? "Saved ✓" : "+ Watchlist"}</button>
    </div>
  );
}

function Section({ title, subtitle, children }) {
  return (
    <section className="section">
      <div className="section-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

function normalizeRecommendation(item) {
  const tmdb = item?.tmdb || {};

  return {
    tmdb_id: tmdb.tmdb_id ?? item?.tmdb_id ?? item?.id ?? null,
    title: tmdb.title || item?.title || "Untitled",
    poster_url: tmdb.poster_url || item?.poster_url || null,
    release_date: tmdb.release_date || item?.release_date || "",
    vote_average: tmdb.vote_average ?? item?.vote_average ?? null,
    similarity: item?.score,
  };
}

async function enrichRecommendation(item) {
  const base = normalizeRecommendation(item);

  if (base.tmdb_id && base.poster_url) {
    return base;
  }

  try {
    const result = await api.search(base.title, 1);
    const movie = (result?.results || []).find((entry) => entry?.title) || result?.results?.[0];

    if (movie) {
      const posterUrl = movie.poster_path
        ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
        : base.poster_url;

      return {
        ...base,
        tmdb_id: base.tmdb_id || movie.id || null,
        title: base.title || movie.title,
        poster_url: posterUrl,
        release_date: base.release_date || movie.release_date || "",
        vote_average: base.vote_average ?? movie.vote_average ?? null,
      };
    }
  } catch {}

  return base;
}

export default function App() {
  const [view, setView] = useState("home");
  const [category, setCategory] = useState("trending");
  const [homeMovies, setHomeMovies] = useState([]);
  const [homeLoading, setHomeLoading] = useState(true);
  const [homeError, setHomeError] = useState("");
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const searchTimer = useRef(null);
  const [details, setDetails] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState("");
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const [recommendationsError, setRecommendationsError] = useState("");
  const [tfidfCards, setTfidfCards] = useState([]);
  const [genreCards, setGenreCards] = useState([]);
  const [profile, setProfile] = useState(() => readProfile());
  const [vibeText, setVibeText] = useState("");
  const [vibeResults, setVibeResults] = useState([]);
  const [vibeLoading, setVibeLoading] = useState(false);
  const [vibeError, setVibeError] = useState("");
  const [vibeSubmitted, setVibeSubmitted] = useState(false);
  const vibeRequestId = useRef(0);
  const [vibePersonalize, setVibePersonalize] = useState(true);
  const [vibeFilters, setVibeFilters] = useState({ genre: "", min_rating: "", max_rating: "", year_from: "", year_to: "", language: "", min_runtime: "", max_runtime: "" });
  const [personalizedMovies, setPersonalizedMovies] = useState([]);

  useEffect(() => saveProfile(profile), [profile]);

  useEffect(() => {
    if (!profile.liked.length) { setPersonalizedMovies([]); return; }
    let active = true;
    api.vibe({ query: "thoughtful, engaging movie stories", personalize: true, liked_ids: profile.liked, disliked_ids: profile.disliked, limit: 12 })
      .then((data) => { if (active) setPersonalizedMovies(data.results || []); })
      .catch(() => { if (active) setPersonalizedMovies([]); });
    return () => { active = false; };
  }, [profile.liked, profile.disliked]);

  useEffect(() => {
    loadHome(category);
  }, [category]);

  useEffect(() => {
    const onPop = (event) => {
      const params = new URLSearchParams(window.location.search);
      const id = params.get("id");
      if (id) {
        openMovie(Number(id), false);
      } else {
        setView(event.state?.view || "home");
        setDetails(null);
      }
    };

    window.addEventListener("popstate", onPop);
    const initialParams = new URLSearchParams(window.location.search);
    const initialId = initialParams.get("id");
    if (initialId) openMovie(Number(initialId), false);
    else if (["discover", "watchlist"].includes(initialParams.get("view"))) setView(initialParams.get("view"));
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  async function loadHome(nextCategory) {
    setHomeLoading(true);
    setHomeError("");

    try {
      const data = await api.home(nextCategory, 24);
      setHomeMovies(Array.isArray(data) ? data : data?.results || []);
    } catch (error) {
      setHomeError(error.message);
    } finally {
      setHomeLoading(false);
    }
  }

  function searchMovies(value) {
    setQuery(value);
    clearTimeout(searchTimer.current);

    if (value.trim().length < 2) {
      setSearchResults([]);
      setSearchError("");
      return;
    }

    searchTimer.current = setTimeout(async () => {
      setSearchLoading(true);
      setSearchError("");

      try {
        const data = await api.search(value.trim(), 1);
        setSearchResults((data?.results || []).filter((movie) => movie?.title).slice(0, 10));
      } catch (error) {
        setSearchError(error.message);
      } finally {
        setSearchLoading(false);
      }
    }, 600);
  }

  async function loadRecommendations(movie) {
    const title = movie?.title;
    const tmdbId = movie?.tmdb_id;

    setRecommendationsLoading(true);
    setRecommendationsError("");
    setTfidfCards([]);
    setGenreCards([]);

    let bundle = null;

    try {
      bundle = await api.bundle(title, 12, 18);
    } catch (error) {
      setRecommendationsError(error.message);
    }

    let localTfidf = bundle?.tfidf_recommendations || [];
    let localGenre = bundle?.genre_recommendations || [];

    if (!localTfidf.length) {
      try {
        localTfidf = await api.tfidf(title, 12);
      } catch {}
    }

    if (!localGenre.length && tmdbId) {
      try {
        localGenre = await api.genreRecommendations(tmdbId, 18);
      } catch {}
    }

    const enrichedTfidf = await Promise.all(
      localTfidf.map((item) => enrichRecommendation(item))
    );

    const validTfidf = enrichedTfidf.filter((movie) => movie.title);
    const validGenre = (localGenre || []).filter((movie) => movie?.title);

    setTfidfCards(validTfidf);
    setGenreCards(validGenre);
    setRecommendationsLoading(false);
  }

  async function openMovie(id, push = true) {
    setView("details");
    setDetails(null);
    setDetailsError("");
    setTfidfCards([]);
    setGenreCards([]);
    setRecommendationsError("");

    if (push) {
      window.history.pushState({}, "", `?view=details&id=${id}`);
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
    setDetailsLoading(true);

    try {
      const movie = await api.details(id);
      setDetails(movie);
      setDetailsLoading(false);
      loadRecommendations(movie);
    } catch (error) {
      setDetailsError(error.message);
      setDetailsLoading(false);
    }
  }

  function goHome() {
    setView("home");
    setDetails(null);
    setQuery("");
    setSearchResults([]);
    setTfidfCards([]);
    setGenreCards([]);
    window.history.pushState({}, "", "/");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function updateFeedback(movie, field) {
    setProfile((current) => toggleMovieFeedback(current, movie, field));
  }

  async function submitVibe(event) {
    event?.preventDefault();
    if (vibeText.trim().length < 3) {
      vibeRequestId.current += 1;
      setVibeSubmitted(true);
      setVibeLoading(false);
      setVibeResults([]);
      setVibeError("Describe the kind of movie you want in at least 3 characters.");
      return;
    }
    const requestId = ++vibeRequestId.current;
    setVibeSubmitted(true); setVibeLoading(true); setVibeError("");
    try {
      const filters = Object.fromEntries(Object.entries(vibeFilters).filter(([, value]) => value !== "").map(([key, value]) => [key, ["min_rating", "max_rating"].includes(key) ? Number(value) : ["year_from", "year_to", "min_runtime", "max_runtime"].includes(key) ? Number(value) : value]));
      const response = await api.vibe({ query: vibeText.trim(), filters, personalize: vibePersonalize, liked_ids: vibePersonalize ? profile.liked : [], disliked_ids: vibePersonalize ? profile.disliked : [] });
      if (requestId === vibeRequestId.current) setVibeResults(response.results || []);
    } catch (error) { if (requestId === vibeRequestId.current) { setVibeError(error.message); setVibeResults([]); } }
    finally { if (requestId === vibeRequestId.current) setVibeLoading(false); }
  }

  const currentCategoryLabel = useMemo(
    () => categories.find(([key]) => key === category)?.[1] || "Movies",
    [category]
  );
  const savedMovie = (id) => profile.moviesById[id] || { tmdb_id: id, title: `Movie ${id}` };

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={goHome}>
          <span className="brand-mark">C</span>
          <span>Cine<span>Match</span></span>
        </button>

        <nav className="desktop-nav">
          <button className={view === "home" ? "nav-active" : ""} onClick={goHome}>Home</button>
          <button className={view === "discover" ? "nav-active" : ""} onClick={() => { setView("discover"); window.history.pushState({ view: "discover" }, "", "?view=discover"); }}>Vibe search</button>
          <button className={view === "watchlist" ? "nav-active" : ""} onClick={() => { setView("watchlist"); window.history.pushState({ view: "watchlist" }, "", "?view=watchlist"); }}>Watchlist ({profile.watchlist.length})</button>
          {categories.slice(0, 3).map(([key, label]) => (
            <button
              key={key}
              className={view === "home" && category === key ? "nav-active" : ""}
              onClick={() => {
                setView("home");
                setCategory(key);
                window.history.pushState({}, "", "/");
              }}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="nav-actions">
          <button className="icon-button" onClick={() => document.querySelector(".hero-search input")?.focus()}>⌕</button>
          <button className="nav-cta" onClick={() => document.querySelector(".hero-search input")?.focus()}>Find a movie</button>
        </div>
      </header>

      {view === "discover" ? (
        <main className="discover-page">
          <span className="eyebrow">VIBE BASED DISCOVERY</span>
          <h1>What are you <em>in the mood for?</em></h1>
          <p className="discover-intro">Describe a feeling, theme, or story. Results use the movie catalog and filters below.</p>
          <form className="vibe-form" onSubmit={submitVibe}>
            <textarea value={vibeText} onChange={(e) => setVibeText(e.target.value)} maxLength={500} placeholder="A mind-bending sci-fi movie with an emotional story..." aria-label="Describe a movie vibe" />
            <div className="prompt-chips">{["A dark psychological thriller with an unpredictable plot", "A relaxing feel-good movie for tonight", "A mystery that slowly builds tension"].map((prompt) => <button type="button" key={prompt} onClick={() => setVibeText(prompt)}>{prompt}</button>)}</div>
            <div className="filter-grid">
              <label>Genre<input value={vibeFilters.genre} onChange={(e) => setVibeFilters({ ...vibeFilters, genre: e.target.value })} placeholder="e.g. Science Fiction" /></label>
              <label>Minimum rating<input type="number" min="0" max="10" step="0.1" value={vibeFilters.min_rating} onChange={(e) => setVibeFilters({ ...vibeFilters, min_rating: e.target.value })} /></label>
              <label>Maximum rating<input type="number" min="0" max="10" step="0.1" value={vibeFilters.max_rating} onChange={(e) => setVibeFilters({ ...vibeFilters, max_rating: e.target.value })} /></label>
              <label>Year from<input type="number" min="1870" max="2100" value={vibeFilters.year_from} onChange={(e) => setVibeFilters({ ...vibeFilters, year_from: e.target.value })} /></label>
              <label>Year to<input type="number" min="1870" max="2100" value={vibeFilters.year_to} onChange={(e) => setVibeFilters({ ...vibeFilters, year_to: e.target.value })} /></label>
              <label>Language code<input maxLength="5" value={vibeFilters.language} onChange={(e) => setVibeFilters({ ...vibeFilters, language: e.target.value })} placeholder="e.g. en" /></label>
              <label>Runtime min<input type="number" min="1" max="1000" value={vibeFilters.min_runtime} onChange={(e) => setVibeFilters({ ...vibeFilters, min_runtime: e.target.value })} /></label>
              <label>Runtime max<input type="number" min="1" max="1000" value={vibeFilters.max_runtime} onChange={(e) => setVibeFilters({ ...vibeFilters, max_runtime: e.target.value })} /></label>
            </div>
            <button type="button" className="text-action" onClick={() => setVibeFilters({ genre: "", min_rating: "", max_rating: "", year_from: "", year_to: "", language: "", min_runtime: "", max_runtime: "" })}>Clear filters</button>
            <label className="toggle-line"><input type="checkbox" checked={vibePersonalize} onChange={(e) => setVibePersonalize(e.target.checked)} /> Include my likes in ranking ({profile.liked.length} liked)</label>
            <button className="primary-button" type="submit" disabled={vibeLoading}>{vibeLoading ? "Finding your movies..." : "Find movies"}</button>
          </form>
          {vibeError && <div className="error-box">{vibeError}</div>}
          {vibeLoading && <div className="recommendation-loading"><span className="loader" /> Searching your movie catalog...</div>}
          {!vibeLoading && vibeSubmitted && !vibeResults.length && !vibeError && <div className="empty-state">No movies matched this vibe and its filters. Try broadening the filters or changing the description.</div>}
          {!!vibeResults.length && <Section title="Your matches" subtitle="Scores rank catalog relevance; they are not probabilities"><div className="movie-grid">{vibeResults.map((movie) => <div key={movie.tmdb_id}><MovieCard movie={movie} score={movie.score} onOpen={openMovie} /><p className="match-explanation">{movie.explanation}</p><MovieFeedback movie={movie} profile={profile} onFeedback={updateFeedback} /></div>)}</div></Section>}
        </main>
      ) : view === "watchlist" ? (
        <main><Section title="Your taste profile" subtitle="Stored in this browser only. Clearing browser storage removes it."><div className="profile-summary"><span>{profile.liked.length} liked</span><span>{profile.disliked.length} hidden</span><span>{profile.watchlist.length} saved</span><button className="text-action" onClick={() => { if (window.confirm("Reset your CineMatch profile on this device? This clears its likes, dislikes, and watchlist.")) setProfile(resetProfile()); }}>Reset profile</button></div></Section><Section title="Your Watchlist" subtitle="Saved on this device">{profile.watchlist.length ? <div className="movie-grid">{profile.watchlist.map((id) => <div key={id}><MovieCard movie={savedMovie(id)} onOpen={openMovie} /><button className="text-action" onClick={() => updateFeedback(savedMovie(id), "watchlist")}>Remove</button></div>)}</div> : <div className="empty-state">Your watchlist is empty. Save movies from Discover or a movie detail page.</div>}</Section><Section title="Movies you like" subtitle="Used only when personalized ranking is enabled"><div className="movie-grid">{profile.liked.map((id) => <div key={id}><MovieCard movie={savedMovie(id)} onOpen={openMovie} /><button className="text-action" onClick={() => updateFeedback(savedMovie(id), "liked")}>Undo like</button></div>)}</div></Section><Section title="Movies hidden from recommendations"><div className="movie-grid">{profile.disliked.map((id) => <div key={id}><MovieCard movie={savedMovie(id)} onOpen={openMovie} /><button className="text-action" onClick={() => updateFeedback(savedMovie(id), "disliked")}>Undo</button></div>)}</div></Section></main>
      ) : view === "home" ? (
        <>
          <main>
            <section className="hero">
              <div className="hero-copy">
                <span className="eyebrow">MOVIE DISCOVERY</span>
                <h1>Find your next<br /><em>favorite</em> movie.</h1>
                <p>Search for a movie you love and discover other movies you may enjoy.</p>

                <div className="hero-search">
                  <span>⌕</span>
                  <input
                    value={query}
                    onChange={(event) => searchMovies(event.target.value)}
                    placeholder="Search for a movie..."
                  />
                  {query && (
                    <button
                      className="clear-search"
                      onClick={() => {
                        setQuery("");
                        setSearchResults([]);
                      }}
                    >
                      ×
                    </button>
                  )}
                  <button
                    className="search-button"
                    onClick={() => searchResults[0] && openMovie(searchResults[0].id)}
                  >
                    Search
                  </button>

                  {query.trim().length >= 2 && (searchLoading || searchResults.length > 0 || searchError) && (
                    <div className="search-dropdown">
                      {searchLoading && <div className="search-state">Searching...</div>}
                      {searchError && <div className="search-state error-text">{searchError}</div>}
                      {!searchLoading && !searchError && searchResults.map((movie) => (
                        <button key={movie.id} className="search-result" onClick={() => openMovie(movie.id)}>
                          {movie.poster_path ? (
                            <img src={`https://image.tmdb.org/t/p/w92${movie.poster_path}`} alt="" />
                          ) : (
                            <div className="mini-poster">🎬</div>
                          )}
                          <span>
                            <strong>{movie.title}</strong>
                            <small>{yearOf(movie.release_date) || "Release date unavailable"}</small>
                          </span>
                          <span className="arrow">→</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="hero-tags">
                  <span>Try</span>
                  {["Interstellar", "Batman", "Avengers"].map((item) => (
                    <button key={item} onClick={() => searchMovies(item)}>{item}</button>
                  ))}
                </div>
              </div>

              <div className="hero-art" aria-hidden="true">
                <div className="art-orbit orbit-one" />
                <div className="art-orbit orbit-two" />
                <div className="art-card card-back" />
                <div className="art-card card-mid" />
                <div className="art-card card-front">
                  <div className="art-poster" />
                  <div className="art-label">MOVIE<br /><b>DISCOVERY</b></div>
                </div>
                <div className="floating-badge badge-top">★ 9.2 <small>Top rated</small></div>
                <div className="floating-badge badge-bottom">✦ <small>Similar picks</small></div>
              </div>
            </section>

            <div className="category-tabs">
              {categories.map(([key, label]) => (
                <button
                  key={key}
                  className={category === key ? "tab-active" : ""}
                  onClick={() => setCategory(key)}
                >
                  {label}
                </button>
              ))}
            </div>

            {!!personalizedMovies.length && <Section title="Recommended for you" subtitle="Ranked using your explicit likes and hidden dislikes"><div className="movie-grid">{personalizedMovies.map((movie) => <div key={movie.tmdb_id}><MovieCard movie={movie} score={movie.score} onOpen={openMovie} /><p className="match-explanation">{movie.explanation}</p><MovieFeedback movie={movie} profile={profile} onFeedback={updateFeedback} /></div>)}</div></Section>}

            <Section title={currentCategoryLabel} subtitle="Fresh movie picks">
              {homeLoading ? (
                <div className="movie-grid">
                  {Array.from({ length: 12 }).map((_, index) => (
                    <div className="skeleton-card" key={index} />
                  ))}
                </div>
              ) : homeError ? (
                <div className="error-box">{homeError}</div>
              ) : homeMovies.length ? (
                <div className="movie-grid">
                  {homeMovies.map((movie) => (
                    <div key={movie.tmdb_id}><MovieCard movie={movie} onOpen={openMovie} /><MovieFeedback movie={movie} profile={profile} onFeedback={updateFeedback} /></div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">No movies available right now.</div>
              )}
            </Section>
          </main>

          <footer>
            <div><span className="brand-mark small">C</span> CineMatch</div>
            <span>Discover · Explore · Watch</span>
          </footer>
        </>
      ) : (
        <main className="details-page">
          <button className="back-button" onClick={goHome}>← Back to movies</button>

          {detailsLoading && !details ? (
            <div className="details-skeleton">
              <div className="detail-poster-skeleton" />
              <div className="detail-copy-skeleton"><div /><div /><div /><div /></div>
            </div>
          ) : detailsError ? (
            <div className="error-box">{detailsError}</div>
          ) : details ? (
            <>
              <section className="detail-hero">
                <div
                  className="backdrop"
                  style={details.backdrop_url ? { backgroundImage: `url(${details.backdrop_url})` } : {}}
                />
                <div className="detail-content">
                  <div className="detail-poster">
                    {details.poster_url ? (
                      <img src={details.poster_url} alt={details.title} />
                    ) : (
                      <div className="poster-empty">🎬</div>
                    )}
                  </div>

                  <div className="detail-info">
                    <span className="eyebrow">MOVIE DETAILS</span>
                    <h1>{details.title}</h1>
                    <div className="detail-meta">
                      <span>{yearOf(details.release_date) || "—"}</span>
                      <span>•</span>
                      <span>{details.genres?.map((genre) => genre.name).join(" · ") || "Genre unavailable"}</span>
                    </div>
                    <p>{details.overview || "No overview available."}</p>
                    <button
                      className="primary-button"
                      onClick={() => document.getElementById("recommendations")?.scrollIntoView({ behavior: "smooth" })}
                    >
                      ✦ Find similar movies
                    </button>
                    <div className="detail-feedback">
                      <button onClick={() => updateFeedback(details, "watchlist")}>{profile.watchlist.includes(details.tmdb_id) ? "In watchlist ✓" : "+ Watchlist"}</button>
                      <button onClick={() => updateFeedback(details, "liked")}>{profile.liked.includes(details.tmdb_id) ? "Liked ✓" : "Like"}</button>
                      <button onClick={() => updateFeedback(details, "disliked")}>{profile.disliked.includes(details.tmdb_id) ? "Hidden ✓" : "Not for me"}</button>
                    </div>
                  </div>
                </div>
              </section>

              <div id="recommendations" className="recommendation-area">
                {recommendationsLoading && (
                  <div className="recommendation-loading">
                    <span className="loader" /> Finding movies you may like...
                  </div>
                )}

                {recommendationsError && !recommendationsLoading && (
                  <div className="recommendation-note">
                    Some recommendations could not be loaded. Showing available results.
                  </div>
                )}

                <Section title="Similar Movies" subtitle="Movies with similar details and themes">
                  {tfidfCards.length ? (
                    <div className="movie-grid">
                      {tfidfCards.map((movie, index) => (
                        <MovieCard
                          key={`${movie.tmdb_id || movie.title}-${index}`}
                          movie={movie}
                          score={movie.similarity}
                          onOpen={openMovie}
                        />
                      ))}
                    </div>
                  ) : !recommendationsLoading ? (
                    <div className="empty-state">No similar movies were found for this title.</div>
                  ) : null}
                </Section>

                <Section title="More Like This" subtitle="Popular movies from the same genre">
                  {genreCards.length ? (
                    <div className="movie-grid">
                      {genreCards.map((movie, index) => (
                        <MovieCard
                          key={`${movie.tmdb_id || movie.title}-${index}`}
                          movie={movie}
                          onOpen={openMovie}
                        />
                      ))}
                    </div>
                  ) : !recommendationsLoading ? (
                    <div className="empty-state">No movies from the same genre were found.</div>
                  ) : null}
                </Section>
              </div>
            </>
          ) : null}
        </main>
      )}
    </div>
  );
}
