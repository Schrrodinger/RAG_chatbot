from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        logger.info("RAG Pipeline initialized successfully")

    def process_query(self, query: str, **kwargs) -> Dict:
        """Process user query through the RAG pipeline."""
        try:
            logger.info(f"Processing query: {query}")

            # Retrieve relevant products
            relevant_products = self.retriever.retrieve_relevant_products(query)
            logger.info(f"Retrieved {len(relevant_products)} relevant products")

            # Generate response
            response = self.generator.generate_response(
                query=query,
                relevant_products=relevant_products  # Changed from context to relevant_products
            )

            result = {
                'response': response,
                'relevant_products': relevant_products,
                'query': query
            }
            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise Exception(f"Lỗi xử lý câu hỏi: {str(e)}")

    def compare_products(self, product_ids: List[str]) -> Dict:
        """Generate product comparison."""
        try:
            products = [p for p in self.retriever.product_data if str(p['id']) in product_ids]
            comparison_query = f"So sánh chi tiết các sản phẩm sau: {', '.join([p['name'] for p in products])}"
            return self.process_query(comparison_query)
        except Exception as e:
            logger.error(f"Error comparing products: {str(e)}")
            raise

    def get_recommendations(self, budget: float, preferences: str) -> Dict:
        """Get product recommendations based on budget and preferences."""
        try:
            query = f"Gợi ý sản phẩm với ngân sách {budget}đ và yêu cầu: {preferences}"
            return self.process_query(query)
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise