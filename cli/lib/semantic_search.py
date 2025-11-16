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
