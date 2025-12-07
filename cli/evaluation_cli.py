import argparse
import json

from lib.hybrid_search import rrf_search_individual


def main():
    parser = argparse.ArgumentParser(description="Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "rb") as f:
        golden_dataset = json.load(f)
        cases = golden_dataset["test_cases"]

    for case in cases:
        print(f"- Query: {case['query']}")
        results = rrf_search_individual(case["query"], 60, limit)
        relevant_retrieved = 0
        total_relevant = len(case["relevant_docs"])
        movies_list = ""
        relevant_list = ", ".join(case["relevant_docs"])
        for idx, result in enumerate(results, start=1):
            movies_list += f"{result['title']}, "
            if result["title"] in case["relevant_docs"]:
                relevant_retrieved += 1

        precision = relevant_retrieved / limit
        recall_precision = relevant_retrieved / total_relevant
        f1_score = 2 * (precision * recall_precision) / (precision + recall_precision)

        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall_precision:.4f}")
        print(f"    - F1 Score: {f1_score:.4f}")
        print(f"    - Retrieved: {movies_list}")
        print(f"    - Relevant: {relevant_list}")
        print("\n")


if __name__ == "__main__":
    main()
