# chatbot/test_rag.py
import json
import logging
from embeddings import DocumentEncoder
from retriever import ProductRetriever
from generator import ResponseGenerator
from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_data():
    """Load sample product data from JSON file"""
    try:
        with open('sample_products.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("sample_products.json not found. Please run sample_data.py first.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing sample_products.json: {str(e)}")
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
    try:
        # Load sample data
        products = load_sample_data()
        logger.info(f"Loaded {len(products)} products")

        if not products:
            logger.error("No products loaded. Exiting test.")
            return

        # Initialize RAG components
        logger.info("Initializing RAG components...")
        encoder = DocumentEncoder()
        retriever = ProductRetriever(encoder)
        generator = ResponseGenerator()
        rag_pipeline = RAGPipeline(retriever, generator)

        # Index products
        logger.info("Indexing products...")
        retriever.index_products(products)

        # Test queries
        test_queries = [
            "Tìm cho tôi điện thoại Samsung tầm giá 10 triệu",
            "So sánh iPhone 14 Pro Max và Samsung Galaxy S23 Ultra",
            "Điện thoại nào có camera tốt nhất trong tầm giá 15 triệu?",
            "Có điện thoại nào đang khuyến mãi không?",
            "Đánh giá pin của Xiaomi 13 Pro như thế nào?"
        ]

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
                preferences="cần camera tốt và pin trâu"
            )
            logger.info(f"Recommendation response: {recommendation_result['response']}")
        except Exception as e:
            logger.error(f"Error in recommendations: {str(e)}")

    except Exception as e:
        logger.error(f"Critical error in test_rag_system: {str(e)}")


if __name__ == "__main__":
    test_rag_system()