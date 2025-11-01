#!/usr/bin/env python3

import argparse
import os
import json
import string

from nltk.stem import PorterStemmer
from inverted_index import inverted_index

cli_dir = os.path.dirname(__file__)
json_path = os.path.join(cli_dir, "../data/movies.json")
stopwords_path = os.path.join(cli_dir, "../data/stopwords.txt")
cache_dir = os.path.join(cli_dir, "../cache")

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)
movies = data["movies"]

_STEMMER = PorterStemmer()

def stemming(text: str, stopwords_path: str) -> list[str]:
    tokens = without_stopwords(text, stopwords_path)
    stemmed_tokens = [_STEMMER.stem(token) for token in tokens]
    return stemmed_tokens

def without_stopwords(text: str, stopwords_file: str) -> list[str]:
    tokens = tokenize(text)

    with open(stopwords_file, "r", encoding="utf-8") as file:
        stopwords = {line.strip().lower() for line in file if line.strip()}

    filtered_tokens = [t for t in tokens if t not in stopwords]
    return filtered_tokens

def tokenize(text: str) -> list[str]:
    cleaned = clean_text(text)
    return cleaned.split()

def clean_text(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).lower()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build and Cache inverted index")
    
    search_parser.add_argument("query", type=str, help="Search query")
    
    args = parser.parse_args()

    match args.command:
        case "build":
            def _tok(text: str):
                return stemming(text, stopwords_path)

            idx = inverted_index(tokenize_fn=_tok)
            idx.build(movies)
            idx.save(cache_dir)

            docs = idx.get_documents("merida")
            if docs:
                print(f"First ID for 'merida': {docs[0]}")
            else:
                print(f"No documents found for 'merida'.")
            
        case "search":
            query_text = stemming(args.query, stopwords_path)
            results = []
    
            for movie in movies:
                titles = stemming(movie["title"], stopwords_path)
                if any(any(q_token in t_token for t_token in titles) for q_token in query_text):
                    results.append(movie["title"])

            if results:
                for index, title in enumerate(results, start=1):
                    print(f"{index}. {title}")
            else:
                print("Result are not found!.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
