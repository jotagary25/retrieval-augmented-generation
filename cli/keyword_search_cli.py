#!/usr/bin/env python3

import argparse
import os
import math
import json

from utils import stemming
from inverted_index import inverted_index

cli_dir = os.path.dirname(__file__)
json_path = os.path.join(cli_dir, "../data/movies.json")
stopwords_path = os.path.join(cli_dir, "../data/stopwords.txt")
cache_dir = os.path.join(cli_dir, "../cache")

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)
movies = data["movies"]

def tok(text: str):
    return stemming(text, stopwords_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build and Cache inverted index")
    tf_parser = subparsers.add_parser("tf", help="show term frequency for a document")
    idf_parser = subparsers.add_parser("idf", help="show inverse document frequency for a term")
    tfidf_parser = subparsers.add_parser("tfidf", help="show TF x IDF for a document")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count in the document")
    idf_parser.add_argument("term", type=str, help="Term to compute IDF for")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to compute TF x IDF for")

    args = parser.parse_args()

    match args.command:
        case "tfidf":
            idx = inverted_index(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except ValueError as e:
                print(f"Error: {e}")
                return

            try:
                tf = idx.get_frequencie(args.doc_id, args.term)
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
            idx = inverted_index(tokenize_fn = tok)
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

            N = len(idx.docmap)
            DF = len(idx.index.get(term_tok, set()))

            if DF == 0 or N == 0:
                idf = 0.0
            else:
                idf = math.log((N + 1) / (DF + 1))
            print(f"{idf:.2f}")

        case "tf":
            idx = inverted_index(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError:
                print("Error: cache not found. Run: uv run cli/keyword_search_cli.py build")
                return

            try:
                tf = idx.get_frequencie(args.doc_id, args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(tf)

        case "build":
            def _tok(text: str):
                return stemming(text, stopwords_path)

            idx = inverted_index(tokenize_fn=_tok)
            idx.build(movies)
            idx.save(cache_dir)

        case "search":
            idx = inverted_index(tokenize_fn = tok)
            try:
                idx.load(cache_dir)
            except FileNotFoundError:
                print("Error: no hay indice en cache. Ejecuta uv run cli/keyword_search.py build")
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
