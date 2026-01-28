import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


class VectorDB:
    def __init__(self):
        self.index = None
        self.documents = []

    def add_texts(self, texts):
        self.documents.extend(texts)
        embeddings = model.encode(texts)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query, k=3):
        if not self.index: return []
        query_embedding = model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.documents[i] for i in indices[0] if i != -1]