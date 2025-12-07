import argparse
import time
import json

from sentence_transformers import CrossEncoder

from lib.hybrid_search import (
    weighted_search,
    rrf_search,
    rrf_search_individual,
)
from test_gemini import (
    enhanced_spell_query,
    enhanced_rewrite_query,
    enhanced_expand_query,
    individual_rerank,
    batch_rerank,
    evaluate_results,
)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the scores to range 0-1"
    )
    weightedSearch_parser = subparsers.add_parser(
        "weighted-search", help="Weighted search"
    )
    rrfSearch_parser = subparsers.add_parser("rrf-search", help="RRF search")

    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="Scores to normalize"
    )
    weightedSearch_parser.add_argument("query", type=str, help="Query to search")
    weightedSearch_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value for weighted search"
    )
    weightedSearch_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    rrfSearch_parser.add_argument("query", type=str, help="Query to search")
    rrfSearch_parser.add_argument(
        "--k", type=int, default=60, help="K weight for RRF search"
    )
    rrfSearch_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    rrfSearch_parser.add_argument(
        "--enhanced",
        type=str,
        choices=["spell", "rewrite", "expand"],
        default="expand",
        help="Query enhancement method",
    )
    rrfSearch_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        default="cross_encoder",
        help="Rerank method",
    )
    rrfSearch_parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the results"
    )

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            query = args.query
            method = args.enhanced
            k = args.k
            if args.rerank_method is not None:
                limit = args.limit * 5
            else:
                limit = args.limit

            if method == "spell":
                response = enhanced_spell_query(query)
            elif method == "rewrite":
                response = enhanced_rewrite_query(query)
            elif method == "expand":
                response = enhanced_expand_query(query)
            elif method is None:
                response = query

            print(f"Enhanced query ({method}): '{query}' -> '{response}'\n")

            if args.rerank_method == "batch":
                results = rrf_search_individual(response, k, limit)
                batch_movies = [
                    {
                        "id": result["id"],
                        "title": result["title"],
                        "description": result["description"],
                    }
                    for result in results
                ]
                reponse_text = batch_rerank(response, batch_movies)
                reranked_ids = json.loads(reponse_text)
                movies_dict = {movie["id"]: movie for movie in results}
                reranked_results = [
                    movies_dict[mid] for mid in reranked_ids if mid in movies_dict
                ]

                for idx, result in enumerate(reranked_results, start=1):
                    print(f"{idx}. {result['title']} - id: {result['id']}")
                    print(f"    Rerank Rank: {idx}")
                    print(
                        f"    BM25 Score: {result['keyword_score']:.4f}, Semantic Score: {result['semantic_score']:.4f}"
                    )
                    print(f"    {result['description'][:100]}...")

            elif args.rerank_method == "individual":
                results = rrf_search_individual(response, k, limit)
                for result in results:
                    score = individual_rerank(
                        response, result["title"], result["description"]
                    )
                    result["score"] = float(score)
                    time.sleep(3)

                results.sort(key=lambda result: result["score"], reverse=True)
                for idx, result in enumerate(results, start=1):
                    print(f"{idx}. {result['title']} - id: {result['id']}")
                    print(f"    Rerank Score: {result['score']:.4f}")
                    print(
                        f"    BM25 Score: {result['keyword_score']:.4f}, Semantic Score: {result['semantic_score']:.4f}"
                    )
                    print(f"    {result['description'][:100]}...")

            elif args.rerank_method == "cross_encoder":
                results = rrf_search_individual(response, k, limit)
                pairs_list = []
                for result in results:
                    pairs_list.append(
                        [response, f"{result['title']} - {result['description']}"]
                    )

                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs_list)
                for index, result in enumerate(results):
                    result["cs-score"] = scores[index]
                results.sort(key=lambda result: result["cs-score"], reverse=True)

                if args.evaluate:
                    formatted_results = []
                    for result in results:
                        text = f"{result['title']} - {result['description']}"
                        formatted_results.append(text)

                    final_content = "\n".join(formatted_results)
                    scores_text = evaluate_results(query, final_content)
                    # print(f"Scores text: {scores_text}")
                    scores = json.loads(scores_text)
                    # print(f"Scores: {scores}")

                    for idx, (result, score) in enumerate(zip(results, scores)):
                        print(f"{idx + 1}. {result['title']}: {score}/3")
                else:
                    for idx, result in enumerate(results, start=1):
                        print(f"{idx}. {result['title']} - id: {result['id']}")
                        print(f"    Cross-Encoder Score: {result['cs-score']:.4f}")
                        print(
                            f"    BM25 Score: {result['keyword_score']:.4f}, Semantic Score: {result['semantic_score']:.4f}"
                        )
                        print(f"    {result['description'][:100]}...")

            else:
                rrf_search(response, k, limit)

        case "weighted-search":
            query = args.query
            alpha = args.alpha
            limit = args.limit

            weighted_search(query, alpha, limit)

        case "normalize":
            scores = args.scores
            min_val = min(scores)
            max_val = max(scores)

            if max_val == min_val:
                for _ in scores:
                    print("* 1.0000")
                return

            for score in scores:
                result = (score - min_val) / (max_val - min_val)
                print(f"* {result:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
