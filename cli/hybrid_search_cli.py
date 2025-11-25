import argparse

from lib.hybrid_search import weighted_search, rrf_search


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

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit
            rrf_search(query, k, limit)

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
