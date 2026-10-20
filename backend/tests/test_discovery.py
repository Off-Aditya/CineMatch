import unittest
import asyncio
import os
import tempfile
from unittest.mock import patch

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app import main


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.DataFrame([
            {"id": 1, "title": "Star Journey", "overview": "space exploration and family emotion", "genres": "Science Fiction, Drama", "release_date": "2014-01-01", "runtime": 140, "original_language": "en", "vote_average": 8.0, "poster_path": None},
            {"id": 2, "title": "Deep Space", "overview": "space mission and distant planets", "genres": "Science Fiction", "release_date": "2001-01-01", "runtime": 100, "original_language": "en", "vote_average": 7.5, "poster_path": None},
            {"id": 3, "title": "Quiet Garden", "overview": "a gentle family comedy", "genres": "Comedy", "release_date": "2018-01-01", "runtime": 90, "original_language": "fr", "vote_average": 6.2, "poster_path": None},
        ])
        cls.vectorizer = TfidfVectorizer()
        cls.matrix = cls.vectorizer.fit_transform(cls.frame.overview)

    def setUp(self):
        main.df = self.frame
        main.tfidf_obj = self.vectorizer
        main.tfidf_matrix = self.matrix
        main.semantic_embeddings = None
        main.embedding_encoder = None

    def test_vibe_ranks_relevant_movies_and_bounds_scores(self):
        request = main.VibeRequest(query="space exploration", limit=3)
        scores, mode = main.vibe_rank(request)
        self.assertEqual(mode, "tfidf-fallback")
        self.assertGreater(scores[0], scores[2])
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_filters_and_dislikes_are_applied(self):
        request = main.VibeRequest(query="space movie", filters={"genre": "Science Fiction", "year_from": 2000}, disliked_ids=[1])
        scores, _ = main.vibe_rank(request)
        self.assertEqual(scores[0], -1)
        self.assertEqual(scores[2], -1)
        self.assertGreaterEqual(scores[1], 0)

    def test_rating_and_runtime_filters(self):
        request = main.VibeRequest(query="family", filters={"min_rating": 7, "max_runtime": 120})
        scores, _ = main.vibe_rank(request)
        self.assertEqual(scores[0], -1)
        self.assertGreaterEqual(scores[1], 0)
        self.assertEqual(scores[2], -1)

    def test_max_rating_and_rating_range_validation(self):
        scores, _ = main.vibe_rank(main.VibeRequest(query="space", filters={"max_rating": 7.6}))
        self.assertLess(scores[0], 0)
        self.assertGreaterEqual(scores[1], 0)
        with self.assertRaises(Exception):
            main.VibeRequest(query="space", liked_ids=[0])
        with self.assertRaises(Exception):
            main.VibeRequest(query="   ")

    def test_pagination_uses_rank_order_not_dataframe_positions(self):
        first = asyncio.run(main.discover_vibe(main.VibeRequest(query="space", limit=1, page=1)))
        second = asyncio.run(main.discover_vibe(main.VibeRequest(query="space", limit=1, page=2)))
        self.assertNotEqual(first.results[0].tmdb_id, second.results[0].tmdb_id)
        self.assertEqual(first.total, second.total)

    def test_semantic_ranking_is_used_when_index_and_encoder_are_available(self):
        class Encoder:
            def get_sentence_embedding_dimension(self): return 3
            def encode(self, _queries, normalize_embeddings=True): return [[1.0, 0.0, 0.0]]
        main.semantic_embeddings = __import__("numpy").array([[1, 0, 0], [0.6, 0.8, 0], [0, 1, 0]], dtype=float)
        main.embedding_encoder = Encoder()
        scores, mode = main.vibe_rank(main.VibeRequest(query="space"))
        self.assertEqual(mode, "sentence-transformer")
        self.assertGreater(scores[0], scores[2])

    def test_unknown_preference_ids_are_ignored_and_unsupported_filter_is_rejected(self):
        scores, _ = main.vibe_rank(main.VibeRequest(query="space", personalize=True, liked_ids=[999999]))
        self.assertTrue((scores >= 0).any())
        main.df = self.frame.drop(columns=["runtime"])
        with self.assertRaises(main.HTTPException) as error:
            main.vibe_rank(main.VibeRequest(query="space", filters={"min_runtime": 60}))
        self.assertEqual(error.exception.status_code, 422)

    def test_personalized_ranking_uses_liked_ids(self):
        baseline, _ = main.vibe_rank(main.VibeRequest(query="family", limit=3))
        personalized, _ = main.vibe_rank(main.VibeRequest(query="family", limit=3, personalize=True, liked_ids=[2]))
        self.assertGreater(personalized[1], baseline[1])

    def test_invalid_query_is_rejected(self):
        with self.assertRaises(Exception):
            main.VibeRequest(query="x")

    def test_missing_search_index_returns_503(self):
        main.tfidf_matrix = None
        with self.assertRaises(main.HTTPException) as error:
            main.vibe_rank(main.VibeRequest(query="anything"))
        self.assertEqual(error.exception.status_code, 503)

    def test_discovery_response_contains_ranked_movies_and_explanations(self):
        result = asyncio.run(main.discover_vibe(main.VibeRequest(query="space exploration", limit=2)))
        self.assertEqual(result.query, "space exploration")
        self.assertEqual(len(result.results), 2)
        self.assertTrue(all(movie.explanation and 0 <= movie.score <= 1 for movie in result.results))

    def test_empty_filter_result_and_invalid_range(self):
        empty = main.VibeRequest(query="space", filters={"genre": "Western"})
        self.assertEqual(asyncio.run(main.discover_vibe(empty)).results, [])
        invalid = main.VibeRequest(query="space", filters={"year_from": 2020, "year_to": 1990})
        with self.assertRaises(main.HTTPException) as error:
            main.vibe_rank(invalid)
        self.assertEqual(error.exception.status_code, 422)

    def test_duplicate_title_metadata_joins_by_overview(self):
        catalog = pd.DataFrame([
            {"title": "Shared Title", "overview": "second story", "genres": "Drama"},
            {"title": "Shared Title", "overview": "first story", "genres": "Comedy"},
        ])
        source = pd.DataFrame([
            {"title": "Shared Title", "overview": "first story", "id": 11, "release_date": "2000-01-01", "runtime": 90, "original_language": "en", "poster_path": "/one.jpg"},
            {"title": "Shared Title", "overview": "second story", "id": 22, "release_date": "2001-01-01", "runtime": 100, "original_language": "fr", "poster_path": "/two.jpg"},
        ])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as file:
            path = file.name
        try:
            source.to_csv(path, index=False)
            with patch.object(main, "CATALOG_CSV_PATH", path):
                joined = main.enrich_catalog_metadata(catalog)
            self.assertEqual(joined.id.tolist(), [22, 11])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
