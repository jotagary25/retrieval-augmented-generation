#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model of the semantic search")
    embedText_parser = subparsers.add_parser("embed_text", help="Embed text using the semantic search model")
    verifyEmbeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings of the semantic search model")
    embedQuery_parser = subparsers.add_parser("embedquery", help="Embed query using the semantic search model")

    embedText_parser.add_argument("text", type=str, help="Text to embed")
    embedQuery_parser.add_argument("query", type=str, help="Query to embed")

    args = parser.parse_args()

    match args.command:
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
