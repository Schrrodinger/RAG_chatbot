# chatbot/test_rag.py
import json
import logging
from embeddings import DocumentEncoder
from retriever import ProductRetriever
from generator import ResponseGenerator
from rag_pipeline import RAGPipeline
import os
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample product data from CSV file in data folder"""
    try:
        # Specify the relative path to the 'data' folder
        file_path = os.path.join(os.path.dirname(__file__), "usage_data.csv")
        products_df = pd.read_csv(file_path, encoding='utf-8')
        return products_df.to_dict(orient='records')
    except FileNotFoundError:
        logger.error(f"{file_path} not found. Please check the file path.")
        return []
    except pd.errors.EmptyDataError as e:
        logger.error(f"Error parsing usage_data.csv: {str(e)}")
        return []

def test_single_query(rag_pipeline, query):
    """Test a single query and handle any errors"""
    try:
        logger.info(f"Testing query: {query}")
        result = rag_pipeline.process_query(query)
        logger.info(f"Response: {result['response']}")
        logger.info("Relevant products:")
        for product in result['relevant_products'][:2]:
            logger.info(f"- {product['name']} (Score: {product.get('relevance_score', 0):.2f})")
        return True
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return False

def test_rag_system():
    # Load the sample data
    products = load_sample_data()

    # Initialize components and pipeline
    encoder = DocumentEncoder()
    retriever = ProductRetriever(encoder)
    generator = ResponseGenerator()
    rag_pipeline = RAGPipeline(retriever, generator)

    # Index the merged data
    retriever.index_products(products)

    # Run the tests
    test_queries = ["Tìm laptop giá rẻ có đánh giá tốt", "So sánh Macbook và các laptop khác"]
    
    logger.info("\nTesting individual queries:")
    success_count = sum(test_single_query(rag_pipeline, query) for query in test_queries)
    logger.info(f"Successfully processed {success_count}/{len(test_queries)} queries")

    # Test product comparison
    logger.info("\nTesting product comparison:")
    try:
        comparison_result = rag_pipeline.compare_products([1, 2])
        logger.info(f"Comparison response: {comparison_result['response']}")
    except Exception as e:
        logger.error(f"Error in product comparison: {str(e)}")

    # Test recommendations
    logger.info("\nTesting recommendations:")
    try:
        recommendation_result = rag_pipeline.get_recommendations(
            budget=10000000,
            preferences="chơi game mượt"
        )
        logger.info(f"Recommendation response: {recommendation_result['response']}")
    except Exception as e:
        logger.error(f"Error in recommendations: {str(e)}")

if __name__ == "__main__":
    test_rag_system()
