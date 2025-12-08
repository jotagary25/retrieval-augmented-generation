import argparse

from lib.hybrid_search import rrf_search_rerank
from test_gemini import (
    generate_answer,
    summarize_movies,
    summarize_citations,
    question_summarize,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a movie description"
    )
    citations_parser = subparsers.add_parser(
        "citations", help="Get a summarize of movies with citations"
    )
    question_parser = subparsers.add_parser(
        "question", help="get a answer to a question"
    )

    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "query", type=str, help="Movie description to summarize"
    )
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    question_parser.add_argument("query", type=str, help="Search query for question")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            k = 60
            limit = 5

            results = rrf_search_rerank(query, k, limit)
            movies = [
                f"title: {result['title']} - description: {result['description']}"
                for result in results
            ]
            answer = generate_answer(query, movies)
            for result in results:
                print(f"- {result['title']}")
            print("\nRAG Response:")
            print(answer)
            print("\n")

        case "summarize":
            query = args.query
            k = 60
            limit = args.limit

            results = rrf_search_rerank(query, k, limit)
            movies = [
                f"title: {result['title']} - description: {result['description']}"
                for result in results
            ]
            summary = summarize_movies(query, movies)
            for result in results:
                print(f"- {result['title']}")
            print("\n LLM Summary:")
            print(summary)
            print("\n")

        case "citations":
            query = args.query
            k = 60
            limit = args.limit

            results = rrf_search_rerank(query, k, limit)
            movies = []
            for idx, result in enumerate(results, start=1):
                movies.append(
                    f"id: {idx} - {result['title']} - {result['description']}"
                )
            answer = summarize_citations(query, movies)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']}")

            print("\n LLM answer:")
            print(answer)
            print("\n")

        case "question":
            query = args.query
            k = 60
            limit = args.limit

            results = rrf_search_rerank(query, k, limit)
            movies = [
                f"title: {result['title']} - description: {result['description']}"
                for result in results
            ]
            answer = question_summarize(query, movies)
            for result in results:
                print(f"- {result['title']}")
            print("\nAnswer:")
            print(answer)
            print("\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
