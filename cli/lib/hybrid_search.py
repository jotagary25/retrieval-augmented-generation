import os
import json
from utils import stemming

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder

current_dir = os.path.dirname(__file__)
stopwords_path = os.path.join(current_dir, "../../data/stopwords.txt")
cache_dir = os.path.join(current_dir, "../../cache")


def tok(text: str):
    return stemming(text, stopwords_path)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(tokenize_fn=tok)
        index_path = os.path.join(cache_dir, "index.pkl")
        if not os.path.exists(index_path):
            self.idx.build(documents)
            self.idx.save(cache_dir)
        else:
            self.idx.load(cache_dir)

    def _bm25_search(self, query, limit):
        return self.idx.bm25_search(query, limit=limit)

    def weighted_search(self, query, alpha, limit=5):
        search_limit = limit * 500
        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search_chunks(query, search_limit)

        bm25_scores = {result[0]: result[1] for result in bm25_results}
        semantic_scores = {
            result["movie_id"]: result["score"] for result in semantic_results
        }

        self._normalize_scores(bm25_scores)
        self._normalize_scores(semantic_scores)

        union_id_scores = set(bm25_scores.keys()) | set(semantic_scores.keys())
        results = []

        for identifier in union_id_scores:
            score_bm25 = bm25_scores.get(identifier, 0.0)
            score_semantic = semantic_scores.get(identifier, 0.0)

            hybrid_score = alpha * score_bm25 + (1 - alpha) * score_semantic
            movie = self.idx.docmap[identifier]
            results.append(
                {
                    "id": identifier,
                    "score": hybrid_score,
                    "title": movie["title"],
                    "description": movie["description"],
                    "keyword_score": score_bm25,
                    "semantic_score": score_semantic,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _normalize_scores(self, scores):
        if not scores:
            return []

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            for key in scores:
                scores[key] = 1.0
            return

        for key, value in scores.items():
            normalized = (value - min_val) / (max_val - min_val)
            scores[key] = normalized

    def rrf_search(self, query, k, limit):
        search_limit = limit * 500
        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search_chunks(query, search_limit)

        bm25_rank = {
            result[0]: (1 / (k + idx))
            for idx, result in enumerate(bm25_results, start=1)
        }
        semantic_rank = {
            result["movie_id"]: (1 / (k + idx))
            for idx, result in enumerate(semantic_results, start=1)
        }

        union_ids = set(bm25_rank.keys()) | set(semantic_rank.keys())
        results = []

        for ids in union_ids:
            rrf_bm25 = bm25_rank.get(ids, 0.0)
            rrf_semantic = semantic_rank.get(ids, 0.0)
            rrf_score = rrf_bm25 + rrf_semantic
            movie = self.idx.docmap[ids]
            results.append(
                {
                    "id": ids,
                    "score": rrf_score,
                    "title": movie["title"],
                    "description": movie["description"],
                    "keyword_score": rrf_bm25,
                    "semantic_score": rrf_semantic,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


def weighted_search(query, alpha, limit=5):
    with open("data/movies.json", "rb") as f:
        documents = json.load(f)["movies"]
    hybrid = HybridSearch(documents)
    results = hybrid.weighted_search(query, alpha, limit)

    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result['title']} - id: {result['id']}")
        print(f"    Hybrid Score: {result['score']:.4f}")
        print(
            f"    BM25: {result['keyword_score']:.4f}, Semantic: {result['semantic_score']:.4f}"
        )
        print(f"    {result['description'][:100]}...")


def rrf_search(query, k, limit):
    with open("data/movies.json", "rb") as f:
        documents = json.load(f)["movies"]
    hybrid = HybridSearch(documents)
    results = hybrid.rrf_search(query, k, limit)

    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result['title']} - id: {result['id']}")
        print(f"    RRF Score: {result['score']:.4f}")
        print(
            f"    BM25: {result['keyword_score']:.4f}, Semantic: {result['semantic_score']:.4f}"
        )
        print(f"    {result['description'][:100]}...")


def rrf_search_individual(query, k, limit):
    with open("data/movies.json", "rb") as f:
        documents = json.load(f)["movies"]

    hybrid = HybridSearch(documents)
    results = hybrid.rrf_search(query, k, limit)

    return results


def rrf_search_rerank(query, k, limit):
    with open("data/movies.json", "rb") as f:
        documents = json.load(f)["movies"]
    hybrid = HybridSearch(documents)
    results = hybrid.rrf_search(query, k, limit)

    pairs_list = []
    for result in results:
        pairs_list.append([query, f"{result['title']} - {result['description']}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs_list)
    for index, result in enumerate(results):
        result["cs-score"] = scores[index]
    results.sort(key=lambda result: result["cs-score"], reverse=True)

    return results

    # for idx, result in enumerate(results, start=1):
    #     print(f"{idx}. {result['title']} - id: {result['id']}")
    #     print(f"    Cross-Encoder Score: {result['cs-score']:.4f}")
    #     print(
    #         f"    BM25 Score: {result['keyword_score']:.4f}, Semantic Score: {result['semantic_score']:.4f}"
    #     )
    #     print(f"    {result['description'][:100]}...")
