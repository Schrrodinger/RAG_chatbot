from typing import Dict, List
from retriever import ProductRetriever
from generator import ResponseGenerator


class RAGPipeline:
    def __init__(self, retriever: ProductRetriever, generator: ResponseGenerator):
        self.retriever = retriever
        self.generator = generator

    def process_query(self, query: str) -> Dict:
        """Process user query through the RAG pipeline."""
        # Retrieve relevant products
        relevant_products = self.retriever.retrieve_relevant_products(query)

        # Generate response
        response = self.generator.generate_response(query, relevant_products)

        return {
            'response': response,
            'relevant_products': relevant_products,
            'query': query
        }

    def compare_products(self, product_ids: List[str]) -> Dict:
        """Generate product comparison."""
        products = [p for p in self.retriever.product_data if p['id'] in product_ids]
        comparison_query = f"So sánh chi tiết các sản phẩm sau: {', '.join([p['name'] for p in products])}"

        return self.process_query(comparison_query)

    def get_recommendations(self, budget: float, preferences: str) -> Dict:
        """Get product recommendations based on budget and preferences."""
        query = f"Gợi ý sản phẩm với ngân sách {budget}đ và yêu cầu: {preferences}"
        return self.process_query(query)