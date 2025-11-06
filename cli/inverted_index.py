import os
import pickle
import math
from collections import Counter
from typing import Callable, TypedDict

BM25_K1 = 1.5
BM25_B = 0.75

class Movie(TypedDict):
    id: int
    title: str
    description: str

class InvertedIndex:
    """
    Indice invertido simple:
        - index token -> {doc_ids}
        - docmap: doc_id -> documento completo
        - _tokenize: funcion que se encarga de tokenizar:
            - los document's, para construir el index
            - la busqueda, para encontrar los doc-id's
    """
    _tokenize: Callable[[str], list[str]]

    def __init__(self, tokenize_fn: Callable[[str], list[str]]):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self._tokenize = tokenize_fn

    # función para tokenizar todo un 'elemento' y guardar sus terminos en index
    def __add_document(self, doc_id: int, text:str) -> None:
        tokens = self._tokenize(text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()


        for tok in tokens:
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)
            self.term_frequencies[doc_id][tok] += 1

        if doc_id not in self.doc_lengths:
            self.doc_lengths[doc_id] = 0
        self.doc_lengths[doc_id] += len(tokens)

    # función que solo busca por el primer termino de una cadena (temporal)
    def get_documents(self, term: str) -> list[int]:
        toks = self._tokenize(term)
        if not toks:
            return []
        tok = toks[0]
        return sorted(self.index.get(tok, set()))

    # función que se encarga de construir todo el indice dado una lista de diccionarios (movies.json)
    def build(self, movies: list[Movie]) -> None:
        """
        Se agrega cada película (title + description) al indice y se llena docmap
        """
        for movie in movies:
            doc_id = int(movie["id"])
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self, cache_dir: str) -> None:
        """
        Se guarda index y docmap en:
           cache/index.pkl
           cache/docmap.pkl
        """
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        frequencies_path = os.path.join(cache_dir, "term_frequencies.pkl")
        doc_lengths_path = os.path.join(cache_dir, "doc_lengths.pkl")

        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self, cache_dir: str) -> None:
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        frequencies_path = os.path.join(cache_dir, "term_frequencies.pkl")
        doc_lengths_path = os.path.join(cache_dir, "doc_lengths.pkl")

        if not all(
            os.path.exists(path)
            for path in [index_path, docmap_path, frequencies_path, doc_lengths_path]
        ):
            raise FileNotFoundError("Cache not found. Run 'build' first.")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_frequency(self, doc_id: int, term: str) -> int:
        tokens = self._tokenize(term)
        if len(tokens) != 1:
            raise ValueError("The 'term' must tokenize to exactly one token.")
        tok = tokens[0]

        counter = self.term_frequencies.get(doc_id)
        if not counter:
            return 0
        return int(counter.get(tok, 0))

    def get_bm25_idf(self, term: str) -> float:
        tokens = self._tokenize(term)
        if len(tokens) != 1:
            raise ValueError("The 'term' must to exactly one token")
        token = tokens[0]

        n = len (self.docmap)
        df = len(self.index.get(token, set()))

        if n == 0:
            return 0.00
        return math.log(((n - df + 0.5) / (df + 0.5)) + 1.0)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        try:
            tf = self.get_frequency(doc_id, term)
        except ValueError as e:
            raise ValueError(f"Error: {e}, Invalid '{term}', this must be exactly one token.")

        avg_doc_length = self.__get_avg_doc_length()
        document_length = self.doc_lengths.get(doc_id, 0)
        if (document_length == 0) or (avg_doc_length == 0):
            print("Document length or average document length is zero.")
            return 0.0
        length_norm = 1 - b + b * (document_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25(self, doc_id: int, term: str, k1: float, b: float) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term, k1, b)
        return idf * tf

    def bm25_search(
        self, query: str, k1: float = BM25_K1, b: float = BM25_B, limit: int = 5
    ) -> list[tuple[int, float]]:
        """Devuelve los top `limit` documentos por puntaje BM25.

        Pasos:
        - Tokeniza la consulta.
        - Reúne candidatos (docs que contienen algún token de la query).
        - Suma BM25(doc, token) por cada token y documento candidato.
        - Ordena descendentemente y devuelve los mejores `limit`.
        """
        tokens = self._tokenize(query)
        if not tokens:
            return []

        candidates: set[int] = set()
        for tok in tokens:
            candidates |= self.index.get(tok, set())
        if not candidates:
            return []

        scores: dict[int, float] = {}
        for doc_id in candidates:
            total = 0.0
            for tok in tokens:
                total += self.bm25(doc_id, tok, k1, b)
            if total > 0.0:
                scores[doc_id] = total

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]
