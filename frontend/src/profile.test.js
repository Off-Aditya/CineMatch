import test from "node:test";
import assert from "node:assert/strict";
import { readProfile, resetProfile, saveProfile, toggleMovieFeedback } from "./profile.js";

const store = new Map();
globalThis.localStorage = {
  getItem: (key) => store.has(key) ? store.get(key) : null,
  setItem: (key, value) => store.set(key, String(value)),
  removeItem: (key) => store.delete(key),
};

test("creates and persists an anonymous ID and ID-only movie arrays", () => {
  const profile = readProfile();
  profile.liked = [42];
  profile.moviesById[42] = { tmdb_id: 42, title: "Example", poster_url: null, release_date: "", vote_average: null };
  saveProfile(profile);
  const restored = readProfile();
  assert.equal(restored.profileId, profile.profileId);
  assert.deepEqual(restored.liked, [42]);
  assert.equal(typeof restored.liked[0], "number");
  assert.equal(restored.moviesById[42].title, "Example");
});

test("repairs malformed storage and migrates v1 object entries", () => {
  store.set("cinematch.profile.v1", "not json");
  const repaired = readProfile();
  assert.equal(repaired.schemaVersion, 2);
  store.set("cinematch.profile.v1", JSON.stringify({
    schemaVersion: 1, profileId: "existing-id",
    liked: [{ tmdb_id: 5, title: "Legacy" }], disliked: [{ tmdb_id: 5, title: "Legacy" }],
    watchlist: [{ tmdb_id: 9, title: "Saved" }],
  }));
  const migrated = readProfile();
  assert.equal(migrated.profileId, "existing-id");
  assert.deepEqual(migrated.liked, []);
  assert.deepEqual(migrated.disliked, [5]);
  assert.deepEqual(migrated.watchlist, [9]);
  assert.equal(migrated.moviesById[9].title, "Saved");
});

test("reset removes only CineMatch profile storage", () => {
  store.set("unrelated-key", "keep me");
  resetProfile();
  assert.equal(store.get("unrelated-key"), "keep me");
  assert.ok(readProfile().profileId);
});

test("likes and dislikes are exclusive, reversible, and watchlist IDs stay unique", () => {
  const movie = { tmdb_id: 77, title: "Preference" };
  let profile = readProfile();
  profile = toggleMovieFeedback(profile, movie, "liked");
  profile = toggleMovieFeedback(profile, movie, "disliked");
  assert.deepEqual(profile.liked, []);
  assert.deepEqual(profile.disliked, [77]);
  profile = toggleMovieFeedback(profile, movie, "watchlist");
  profile = toggleMovieFeedback(profile, movie, "watchlist");
  assert.deepEqual(profile.watchlist, []);
  profile = toggleMovieFeedback(profile, movie, "disliked");
  assert.deepEqual(profile.disliked, []);
});
