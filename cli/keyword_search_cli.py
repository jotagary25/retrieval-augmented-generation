#!/usr/bin/env python3

import argparse
import os
import json

cli_dir = os.path.dirname(__file__)
json_path = os.path.join(cli_dir, "../data/movies.json")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                
            movies = data["movies"]

            query_lower = args.query.lower()
            results = [
                movie["title"]
                for movie in movies
                if query_lower in movie["title"].lower()
            ]

            if results:
                for index, title in enumerate(results, start=1):
                    print(f"{index}. {title}")
            else:
                print("Result are not found!.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
