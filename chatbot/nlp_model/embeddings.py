import faiss
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import signal
import sys

class DocumentEncoder:
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        self.batch_size = 16  # Define batch size for encoding

    def encode_documents(self, documents: list) -> np.ndarray:
        """Encode a list of documents in smaller batches."""
        all_embeddings = []
        for i in tqdm(range(0, len(documents), self.batch_size)):
            batch_documents = documents[i:i + self.batch_size]
            embeddings = self.model.encode(batch_documents, convert_to_tensor=True)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into an embedding."""
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        return query_embedding

class ProductRetriever:
    def __init__(self, encoder: DocumentEncoder):
        self.encoder = encoder
        self.index = None
        self.documents = None  # Store documents for later retrieval

    def index_products(self, products: list):
        """Index product data for efficient retrieval using an optimized FAISS index."""
        # Store documents for later use
        self.documents = products
        
        embeddings = self.encoder.encode_documents(products)

        # Set up FAISS index with IndexIVFFlat
        d = embeddings.shape[1]  # Dimensionality of embeddings
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatL2(d)  # Quantizer for coarse quantization
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # Train the index
        self.index.train(embeddings)
        self.index.add(embeddings)

        # Verify the index is populated
        print(f"Indexed {self.index.ntotal} products.")

    def retrieve_relevant_products(self, query: str, k: int = 5) -> list:
        """Retrieve relevant products based on query."""
        if not self.index or self.documents is None:
            raise ValueError("FAISS index has not been initialized. Call index_products() first.")

        # Encode the query
        query_embedding = self.encoder.encode_query(query)

        # Perform the search
        distances, indices = self.index.search(query_embedding, k)
        relevant_products = [self.documents[idx] for idx in indices[0]]
        return relevant_products

# Graceful shutdown handler and example usage remain the same