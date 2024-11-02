import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch


class DocumentEncoder:
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        """Initialize the encoder with a Vietnamese-specific SBERT model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode a list of documents into embeddings."""
        return self.model.encode(documents, convert_to_tensor=True)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into an embedding."""
        return self.model.encode(query, convert_to_tensor=True)

    def build_index(self, documents: List[str]):
        """Build FAISS index from documents."""
        self.documents = documents
        embeddings = self.encode_documents(documents)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def retrieve_similar(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        """Retrieve k most similar documents for a query."""
        query_embedding = self.encode_documents([query])
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((idx, distance, self.documents[idx]))

        return results