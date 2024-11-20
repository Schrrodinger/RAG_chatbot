from typing import Dict, List
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        logger.info("RAG Pipeline initialized successfully")

    def _ensure_numpy(self, embedding):
        """Ensure the embedding is a NumPy array."""
        if torch.is_tensor(embedding):
            return embedding.cpu().numpy()
        return embedding

    def process_query(self, query: str, history: List[Dict] = None, **kwargs) -> Dict:
        """Process user query through the RAG pipeline."""
        try:
            # Retrieve relevant products
            relevant_products = self.retriever.retrieve_relevant_products(query)
            logger.info(f"Retrieved {len(relevant_products)} relevant products")

            # Generate response
            response = self.generator.generate_response(
                query=query,
                relevant_products=relevant_products,
                conversation_history=history
            )

            return {
                'response': response,
                'relevant_products': relevant_products,
                'query': query
            }

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise

    def compare_products(self, product_ids: List[int]) -> Dict:
        """Compare multiple products by their IDs."""
        try:
            comparison = []
            for product_id in product_ids:
                if 0 <= product_id < len(self.retriever.product_data):
                    product = self.retriever.product_data[product_id]
                    comparison.append({
                        'Tên sản phẩm': product.get('name', 'Unknown Product'),
                        'Giá': product.get('price', 'N/A'),
                        'Mô tả': product.get('description', 'N/A'),
                        'RAM': product.get('ram', 'N/A'),
                        'Bộ nhớ': product.get('storage', 'N/A'),
                        'CPU': product.get('processor', 'N/A')
                    })
            return {'response': comparison}
        except Exception as e:
            logger.error(f"Error in product comparison: {str(e)}")
            raise

    def get_recommendations(self, budget: float, preferences: str) -> Dict:
        """Provide product recommendations based on a budget and preferences."""
        try:
            relevant_products = []
            for product in self.retriever.product_data:
                price = float(product.get('price', 0))
                if price <= budget and preferences.lower() in product.get('description', '').lower():
                    relevant_products.append(product)

            sorted_recommendations = sorted(
                relevant_products,
                key=lambda x: float(x.get('price', 0))
            )

            if not sorted_recommendations:
                return {'response': 'Không có sản phẩm nào phù hợp với yêu cầu của bạn.'}

            response = [f"Đây là các sản phẩm phù hợp với ngân sách {budget:,}đ và yêu cầu '{preferences}':"]
            for product in sorted_recommendations[:5]:
                response.append(f"\n• {product['name']} - Giá: {self.generator.format_price(product['price'])}đ")

            return {'response': "\n".join(response)}
        except Exception as e:
            logger.error(f"Error in recommendations: {str(e)}")
            raise