from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        logger.info("RAG Pipeline initialized successfully")

    def process_query(self, query: str, history: List[Dict] = None, **kwargs) -> Dict:
        """Process user query through the RAG pipeline."""
        try:
            logger.info(f"Processing query: {query}")

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
            # Re-raise the exception instead of returning an error message
            raise