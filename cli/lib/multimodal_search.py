import json
from PIL import Image
from sentence_transformers import SentenceTransformer, util


class MultimodalSearch:
    def __init__(self, documents=None, model_name="clip-ViT-B-32"):
        self.documents = documents or []
        self.model = SentenceTransformer(model_name)

        if self.documents:
            self.texts = [
                f"{doc['title']}: {doc['description']}" for doc in self.documents
            ]
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        image = Image.open(image_path)
        return self.model.encode(image)

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        hits = util.semantic_search(image_embedding, self.text_embeddings, top_k=5)[0]

        results = []
        for hit in hits:
            idx = hit["corpus_id"]
            doc = self.documents[idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "score": hit["score"],
                }
            )
        return results


def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    with open("data/movies.json", "rb") as f:
        documents = json.load(f)["movies"]

    search = MultimodalSearch(documents)
    return search.search_with_image(image_path)
