from typing import List, Dict
import pandas as pd
from embeddings import DocumentEncoder
import torch
import faiss
import re
import string
import logging
import numpy as np
from utils import sanitize_for_json

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ProductRetriever:
    def __init__(self, encoder: DocumentEncoder):
        self.encoder = encoder
        self.product_data = None
        self.index = None

    @staticmethod
    def load_data(file_path: str):
        data = pd.read_csv(file_path)
        # Replace NaN or infinite values
        data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
        return data.to_dict(orient='records')
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Perform text preprocessing such as lowering case, removing punctuation, etc."""
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def prepare_product_documents(self, products: List[Dict]) -> List[str]:
        """Convert product data to searchable documents."""
        documents = []
        for product in products:
            # Log a warning if key fields are missing
            if 'name' not in product or 'description' not in product:
                logger.warning(f"Product missing essential fields: {product}")

            # Prepare document
            doc = f"""
            Tên sản phẩm: {product.get('name', '')}
            Mô tả: {product.get('description', '')}
            Giá: {product.get('price', '')}
            Giá ưu đãi: {product.get('special_price', '')}
            RAM: {product.get('ram', '')}
            Bộ nhớ: {product.get('storage', '')}
            CPU: {product.get('processor', '')}
            Hình ảnh: {product.get('primary_image', '')}
            Câu hỏi liên quan: {product.get('question', '')}
            Trả lời: {product.get('answer', '')}
            % Giảm giá: {product.get('discount_percentage', '')}
            """
            documents.append(doc.strip())
        return documents

    def _ensure_numpy(self, embeddings):
        """Ensure the embeddings are a NumPy array."""
        if torch.is_tensor(embeddings):
            return embeddings.cpu().numpy()
        return embeddings

    def index_products(self, products: List[Dict]):
        """Index product data for retrieval."""
        # Sanitize the product data to ensure no non-JSON-compliant values
        sanitized_products = [sanitize_for_json(product) for product in products]

        self.product_data = sanitized_products
        documents = self.prepare_product_documents(sanitized_products)

        embeddings = self.encoder.encode_documents(documents)
        embedding_array = self._ensure_numpy(embeddings)

        # Create FAISS index
        self.index = faiss.IndexFlatL2(embedding_array.shape[1])
        self.index.add(embedding_array)

        logger.info(f"Embedding shape: {embedding_array.shape}, Indexed {self.index.ntotal} embeddings")

    def retrieve_relevant_products(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant products based on query."""
        if not self.index:
            raise ValueError("FAISS index has not been initialized. Call index_products() first")

        # Encode the query and ensure it's a NumPy array
        query_embedding = self.encoder.encode_query(query)
        query_embedding = self._ensure_numpy(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)  # Ensure shape is correct

        # Perform the search and handle output correctly
        distances, indices = self.index.search(query_embedding, k)

        relevant_products = []
        for idx in indices[0]:
            if 0 <= idx < len(self.product_data):
                relevant_products.append(self.product_data[idx])

        return relevant_products