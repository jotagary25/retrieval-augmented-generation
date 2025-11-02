import os
import pickle
from collections import Counter
from typing import Callable, Dict, Set, List

class inverted_index:
    """
    Indice invertido simple:
        - index token -> {doc_ids}
        - docmap: doc_id -> documento completo
        - _tokenize: funcion que se encarga de tokenizar:
            - los document's, para construir el index
            - la busqueda, para encontrar los doc-id's
    """
    def __init__(self, tokenize_fn: Callable[[str], List[str]]):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}
        self.term_frequencies: Dict[int, Counter] = {}
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

    # función que solo busca por el primer termino de una cadena (temporal)
    def get_documents(self, term: str) -> List[int]:
        toks = self._tokenize(term)
        if not toks:
            return []
        tok = toks[0]
        return sorted(self.index.get(tok, set()))

    # función que se encarga de construir todo el indice dado una lista de diccionarios (movies.json)
    def build(self, movies: List[dict]) -> None:
        """
        Se agrega cada película (title + description) al indice y se llena docmap
        """
        for movie in movies:
            doc_id = int(movie["id"])
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    #
    def save(self, cache_dir: str) -> None:
        """
        Se guarda index y docmap en:
           cache/index.pkl
           cache/docmal.pkl
        """
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        frequencies_path = os.path.join(cache_dir, "term_frequencies.pkl")

        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self, cache_dir: str) -> None:
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        frequencies_path = os.path.join(cache_dir, "term_frequencies.pkl")

        if not (os.path.exists(index_path) and os.path.exists(docmap_path) and os.path.exists(frequencies_path)):
            raise FileNotFoundError("Cache not found. Run 'build' first.")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_frequencie(self, doc_id: int, term: str) -> int:
        tokens = self._tokenize(term)
        if len(tokens) != 1:
            raise ValueError("The 'term' must tokenize to exactly on token.")
        tok = tokens[0]

        counter = self.term_frequencies.get(doc_id)
        if not counter:
            return 0
        return int(counter.get(tok, 0))
