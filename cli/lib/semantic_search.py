import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text_list = [text]
        embedding = self.model.encode(text_list)
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        list_resume = []
        for document in documents:
            self.document_map[document["id"]] = document
            list_resume.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(list_resume, show_progress_bar=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for document in documents:
            self.document_map[document["id"]] = document
        cache_path = 'cache/movie_embeddings.npy'
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        similarities = []
        for idx, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((idx, score))

        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]
        results = []
        for similar in sorted_similarities:
            results.append(
                {
                    "score": similar[1],
                    "title": self.documents[similar[0]]["title"],
                    "description": self.documents[similar[0]]["description"]
                }
            )
        return results

def verify_model():
    semantic = SemanticSearch()
    print(f"Model loaded: {semantic.model}")
    print(f"Max sequence length: {semantic.model.max_seq_length}")

def embed_text(text):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic = SemanticSearch()
    with open('data/movies.json', 'rb') as f:
        documents = json.load(f)['movies']
        embeddings = semantic.load_or_create_embeddings(documents)
        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape[0]}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def load_and_search(query, limit):
    semantic = SemanticSearch()
    with open('data/movies.json', 'rb') as f:
        documents = json.load(f)['movies']
        embeddings = semantic.load_or_create_embeddings(documents)

    return semantic.search(query, limit)
