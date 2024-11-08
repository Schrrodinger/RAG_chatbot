from typing import List, Dict
import pandas as pd
from embeddings import DocumentEncoder


class ProductRetriever:
    def __init__(self, encoder: DocumentEncoder):
        self.encoder = encoder
        self.product_data = None

    def prepare_product_documents(self, products: List[Dict]) -> List[str]:
        """Convert product data to searchable documents."""
        documents = []
        for product in products:
            doc = f"""
            Tên sản phẩm: {product['name']}
            Giá: {product['price']}
            Mô tả: {product['description']}
            Thông số kỹ thuật: {product['specifications']}
            Khuyến mãi: {product['promotions']}
            Đánh giá: {product['reviews']}
            Bảo hành: {product['warranty_info']}
            """
            documents.append(doc.strip())
        return documents

    def index_products(self, products: List[Dict]):
        """Index product data for retrieval."""
        self.product_data = products
        documents = self.prepare_product_documents(products)
        self.encoder.build_index(documents)

    def retrieve_relevant_products(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant products based on query."""
        similar_docs = self.encoder.retrieve_similar(query, k)
        relevant_products = []

        for idx, distance, _ in similar_docs:
            product = self.product_data[idx]
            product['relevance_score'] = float(1 / (1 + distance))
            relevant_products.append(product)

        return relevant_products