#!/usr/bin/env python3

import argparse
import os
import math
import json

from utils import stemming
from inverted_index import InvertedIndex

cli_dir = os.path.dirname(__file__)
json_path = os.path.join(cli_dir, "../data/movies.json")
stopwords_path = os.path.join(cli_dir, "../data/stopwords.txt")
cache_dir = os.path.join(cli_dir, "../cache")
BM25_K1 = 1.5
BM25_B = 0.75

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)
movies = data["movies"]

def tok(text: str):
    return stemming(text, stopwords_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies")
    build_parser = subparsers.add_parser("build", help="Build and Cache inverted index")
    tf_parser = subparsers.add_parser("tf", help="show term frequency for a document")
    idf_parser = subparsers.add_parser("idf", help="show inverse document frequency for a term")
    tfidf_parser = subparsers.add_parser("tfidf", help="show TF x IDF for a document")
    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given Document ID and term")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count in the document")
    idf_parser.add_argument("term", type=str, help="Term to compute IDF for")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to compute TF x IDF for")
    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="tunable BM25 k1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="tunable BM25 b parameter")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("k1", type=int, nargs='?', default=BM25_K1, help="tunable BM25 k parameter")
    bm25search_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="tunable BM25 b parameter")
    bm25search_parser.add_argument("--limit", "-1", type=int, nargs='?', default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "bm25search":
            idx = InvertedIndex(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            results = idx.bm25_search(args.query, args.k1, args.b, args.limit)
            if not results:
                print("No results found")
                return

            for i, (doc_id, score) in enumerate(results, start=1):
                movie = idx.docmap.get(doc_id)
                title = movie["title"] if movie else f"Doc {doc_id}"
                print(f"{i}. {title} (ID: {doc_id}) - {score:.2f}")

        case "bm25tf":
            idx = InvertedIndex(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            try:
                bm25tf = idx.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(f"BM25 TF score for '{args.term}' in doc '{args.doc_id}': {bm25tf:.2f}")

        case "bm25idf":
            idx = InvertedIndex(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            try:
                bm25idf = idx.get_bm25_idf(args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(f"BM25 IDF score for {args.term}: {bm25idf:.2f}")

        case "tfidf":
            idx = InvertedIndex(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            try:
                tf = idx.get_frequency(args.doc_id, args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            term_tokens = stemming(args.term, stopwords_path)
            if len(term_tokens) != 1:
                print("Error: the term must be a single word")
                return
            term_tok = term_tokens[0]

            N = len(idx.docmap)
            DF = len(idx.index.get(term_tok, set()))
            idf = math.log((N + 1) / (DF + 1))

            score = tf * idf
            print(f"TF-IDF score for '{args.term}' in document '{args.doc_id}': {score:.2f}")

        case "idf":
            idx = InvertedIndex(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError:
                print("Error: cache not found. Run: uv run cli/keyword_search_cli.py build")
                return

            tokens = stemming(args.term, stopwords_path)
            if len(tokens) != 1:
                print("Error: the term must be a single word")
                return
            term_tok = tokens[0]

            n = len(idx.docmap)
            df = len(idx.index.get(term_tok, set()))

            if df == 0 or n == 0:
                idf = 0.0
            else:
                idf = math.log((n + 1) / (df + 1))
            print(f"{idf:.2f}")

        case "tf":
            idx = InvertedIndex(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError:
                print("Error: cache not found. Run: uv run cli/keyword_search_cli.py build")
                return

            try:
                tf = idx.get_frequency(args.doc_id, args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(tf)

        case "build":
            def _tok(text: str):
                return stemming(text, stopwords_path)

            idx = InvertedIndex(tokenize_fn=_tok)
            idx.build(movies)
            idx.save(cache_dir)

        case "search":
            idx = InvertedIndex(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError:
                print("Error: no hay indice en cache. Ejecuta uv run cli/keyword_search_cli.py build")
                return

            query_tokens = stemming(args.query, stopwords_path)
            seen = set()
            results = []

            for qtok in query_tokens:
                doc_ids = idx.get_documents(qtok)
                for doc_id in doc_ids:
                    if doc_id not in seen:
                        seen.add(doc_id)
                        results.append(doc_id)
                        if len(results) == 5:
                            break
                if len(results) == 5:
                    break

            if not results:
                print("No se encontraron resultados")
            else:
                for i, doc_id in enumerate(results, start=1):
                    title = idx.docmap[doc_id]["title"]
                    print(f"{i}. {title} (ID: {doc_id})")

#         case "search":
#             query_text = stemming(args.query, stopwords_path)
#             results = []
#
#             for movie in movies:
#                 titles = stemming(movie["title"], stopwords_path)
#                 if any(any(q_token in t_token for t_token in titles) for q_token in query_text):
#                     results.append(movie["title"])
#
#             if results:
#                 for index, title in enumerate(results, start=1):
#                     print(f"{index}. {title}")
#             else:
#                 print("Result are not found!.")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
