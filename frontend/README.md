# CineMatch React Frontend

React/Vite frontend for the FastAPI movie recommender.

## Run

1. Copy `.env.example` to `.env`.
2. Set `VITE_API_BASE_URL=http://127.0.0.1:8000` for local FastAPI.
3. From this folder:

```bash
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Backend endpoints used

- `GET /health`
- `GET /home`
- `GET /tmdb/search`
- `GET /movie/id/{tmdb_id}`
- `GET /movie/search`
- `GET /recommend/genre`

The UI uses TMDB posters/details through the FastAPI backend and displays both movie matching similarity recommendations and genre recommendations.
