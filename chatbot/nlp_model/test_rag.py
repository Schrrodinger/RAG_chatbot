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


def evaluate_precision(retriever, queries_ground_truth, k=3):
    """
    Evaluate Precision@1 and Precision@k for retrieval results.

    Args:
        retriever: The retrieval system (e.g., rag_pipeline.retriever).
        queries_ground_truth: List of queries and their expected ground truth results, e.g.,
                              [{'query': '...', 'ground_truth': [list of expected product IDs]}].
        k: The value of k for Precision@k.
    Returns:
        precision_scores: Dictionary containing Precision@1 and Precision@k.
    """
    precision_top1 = []
    precision_topk = []

    for item in queries_ground_truth:
        query = item['query']
        ground_truth = set(item['ground_truth'])  # Convert ground truth to a set for fast lookup

        # Retrieve results using FAISS + Qwen re-ranking
        retrieved_products = retriever.retrieve_relevant_products(query, k=max(1, k))
        re_ranked_products = retriever.re_rank_with_qwen(query, retrieved_products)

        # Extract product IDs from re-ranked results
        retrieved_ids = [product['id'] for product in re_ranked_products]

        # Calculate Precision@1
        if len(retrieved_ids) > 0 and retrieved_ids[0] in ground_truth:
            precision_top1.append(1)
        else:
            precision_top1.append(0)

        # Calculate Precision@k
        relevant_count = sum(1 for product_id in retrieved_ids[:k] if product_id in ground_truth)
        precision_topk.append(relevant_count / k)

    return {
        'Precision@1': sum(precision_top1) / len(precision_top1),
        f'Precision@{k}': sum(precision_topk) / len(precision_topk),
    }


def test_precision_with_qwen(rag_pipeline, queries_ground_truth):
    """
    Test and print Precision@1 and Precision@3 after Qwen integration.
    """
    precision_result = evaluate_precision(rag_pipeline.retriever, queries_ground_truth, k=3)
    logger.info(f"Precision@1: {precision_result['Precision@1']:.2f}")
    logger.info(f"Precision@3: {precision_result['Precision@3']:.2f}")


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

    # Run individual query tests
    test_queries = ["Tìm laptop giá rẻ có đánh giá tốt", "So sánh Macbook và các laptop khác"]
    logger.info("\nTesting individual queries:")
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        result = rag_pipeline.process_query(query)
        logger.info(f"Response: {result['response']}")
        logger.info("Relevant products:")
        for product in result['relevant_products'][:2]:
            logger.info(f"- {product['name']}")

    # Precision evaluation
    queries_ground_truth = [
        {'query': 'Tìm laptop chơi game tốt', 'ground_truth': [1, 2, 3]},
        {'query': 'Laptop giá rẻ', 'ground_truth': [4, 5, 6]},
    ]
    test_precision_with_qwen(rag_pipeline, queries_ground_truth)


if __name__ == "__main__":
    test_rag_system()
