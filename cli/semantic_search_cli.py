#!/usr/bin/env python3

import argparse
import re

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    load_and_search,
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
    search_parser = subparsers.add_parser("search", help="Search using the semantic search model")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk text using the semantic model")
    semanticChunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text using the semantic model")

    embedText_parser.add_argument("text", type=str, help="Text to embed")
    embedQuery_parser.add_argument("query", type=str, help="Query to embed")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, help="Overlap between chunks")
    semanticChunk_parser.add_argument("text", type=str, help="Text to chunk")
    semanticChunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Size of each chunk")
    semanticChunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    args = parser.parse_args()

    match args.command:
        case "semantic_chunk":
            text = args.text
            limit = args.max_chunk_size
            overlap = args.overlap

            regex = r"(?<=[.!?])\s+"
            sentences = re.split(regex, text)
            chunks = []

            for i in range(0, len(sentences), limit):
                if overlap > 0 and i >= overlap:
                    chunk = " ".join(sentences[i-overlap:i+limit])
                else:
                    chunk = " ".join(sentences[i:i+limit])
                chunks.append(chunk)

            print(f"Semantically chunking {len(text)} characters")
            for i in range(len(chunks)):
                print(f"{i+1}. {chunks[i]}")

        case "chunk":
            text = args.text
            limit = args.chunk_size
            overlap = args.overlap
            chunks = text.split(" ")
            list_chunks = []
            print(f"Chunking {len(text)} characters")
            for i in range(0, len(chunks), limit):
                if overlap > 0 and i >= overlap:
                    chunk = " ".join(chunks[i-overlap:i+limit])
                else:
                    chunk = " ".join(chunks[i:i+limit])
                list_chunks.append(chunk)

            for i in range(len(list_chunks)):
                print(f"{i+1}. {list_chunks[i]}")

        case "search":
            results = load_and_search(args.query, limit=args.limit)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']} (score: {result['score']})")
                print(f"{result['description']}")
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
