import faiss
import torch
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import traceback


class DocumentEncoder:
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        self.batch_size = 16
        self.index = None
        self.products = None
        self.embeddings = None

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents in batches."""
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

    def create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS index for efficient similarity search."""
        d = embeddings.shape[1]
        nlist = min(100, len(embeddings) // 10)  # Dynamic cluster number
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        self.index.train(embeddings)
        self.index.add(embeddings)

    def generate_embeddings(self, products: List[str],
                             products_file: str = 'test_products.json',
                             embeddings_file: str = 'product_embeddings.json') -> Dict[str, Any]:
        """Generate and save embeddings."""
        # Generate embeddings
        embeddings = self.encode_documents(products)

        # Create FAISS index
        self.create_faiss_index(embeddings)
        self.products = products

        # Save results
        result = {
            'products': products,
            'embeddings': embeddings.tolist()
        }

        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)

        return result

    def search(self, query: str, top_k: int = 5, expected_top: int = 3) -> List[str]:
        """
        Perform enhanced semantic search, ensuring more precision for top-k results.
        Specifically enhances the expected_top by including products similar to top-1.
        
        Args:
            query (str): The query string to search for.
            top_k (int): Number of top results to return. Default is 5.
            expected_top (int): Number of expected top results to focus on refining for accuracy.
        
        Returns:
            List[str]: List of top-k products.
        """
        if self.index is None or self.products is None:
            raise ValueError("Index not initialized. Call generate_embeddings first.")

        # Encode the query
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.detach().cpu().numpy()

        # Search for the top-k matches
        distances, indices = self.index.search(query_embedding, top_k)
        top_products = [self.products[int(idx)] for idx in indices[0]]

        # Enhance the expected_top results by considering only the top-1 as an anchor
        top_1_product = top_products[0]
        top_1_embedding = self.model.encode([top_1_product], convert_to_tensor=True).detach().cpu().numpy()

        refined_indices, refined_distances = self.index.search(top_1_embedding, top_k + 1)
        refined_products = [self.products[int(idx)] for idx in refined_indices[0] if int(idx) not in indices[0]][:expected_top - 1]

        # Combine the top-1 product with the refined products
        enhanced_results = [top_1_product] + refined_products + top_products[expected_top:]

        return enhanced_results[:top_k]

    def evaluate_embedding_accuracy(self) -> Dict[str, Any]:
        """Evaluate embedding accuracy."""
        if self.index is None or self.products is None:
            raise ValueError("Index not initialized.")

        embeddings = np.array(self.model.encode(self.products, convert_to_tensor=True).detach().cpu().numpy())

        accuracy_results = []
        for i, query_product in enumerate(self.products):
            distances, indices = self.index.search(embeddings[i:i + 1], 6)  # 6 to exclude self
            top_products = [self.products[int(idx)] for idx in indices[0][1:]]  # Exclude first (exact match)

            accuracy_results.append({
                'query_product': query_product,
                'top_similar_products': top_products
            })

        return {
            'total_products': len(self.products),
            'accuracy_results': accuracy_results
        }

    def load_data(self, products: List[str]):
        """Load data and create embeddings."""
        self.products = products
        self.embeddings = self.encode_documents(products)
        self.create_faiss_index(self.embeddings)

    def calculate_precision(self, top_k: int = 3) -> Dict[str, Any]:
        """Calculate Precision."""
        precisions = []

        for i, query_product in enumerate(self.products):
            # Search for top_k+1 to exclude the exact match
            distances, indices = self.index.search(self.embeddings[i:i + 1], top_k + 1)

            # Exclude the first result (exact match)
            similar_products = [self.products[int(idx)] for idx in indices[0][1:]]

            # Determine the base category
            base_category = self._extract_base_category(query_product)

            # Count number of products in the same category
            matches = sum(
                1 for prod in similar_products
                if self._extract_base_category(prod) == base_category
            )

            # Calculate precision
            precision = matches / top_k
            precisions.append(precision)

        return {
            f'Precision@{top_k}': np.mean(precisions),
            'Individual Precisions': precisions
        }

    def _extract_base_category(self, product: str) -> str:
        """Extract the base category."""
        categories = [
            "áo thun", "quần jean", "giày", "áo sơ mi",
            "áo khoác", "quần short", "mũ", "túi", "kính", "đồng hồ"
        ]
        for category in categories:
            if category in product.lower():
                return category
        return product


def create_variations(base_product):
    """Create variations of the base product."""
    return [
        base_product,
        base_product + " cao cấp",
        base_product + " phong cách",
        base_product + " mới nhất",
        base_product + " giá rẻ"
    ]


def main():
    base_products = [
        "Áo thun nam màu trắng",
        "Áo thun nam màu đen",
        "Áo thun nam cổ tròn",
        "Áo thun nam tay ngắn",
        "Quần jean nam xanh đậm",
        "Quần jean nam đen",
        "Quần jean nam skinny",
        "Giày thể thao nam Nike",
        "Giày thể thao nam Adidas",
        "Mũ lưỡi trai nam",
        "Áo khoác gió nam",
        "Áo sơ mi nam trắng",
        "Áo sơ mi nam đen",
        "Áo len nam",
        "Quần short nam",
        "Dây nịt nam da",
        "Vớ nam",
        "Túi đeo chéo nam",
        "Kính mát nam",
        "Đồng hồ nam"
    ]

    test_products = []
    for base_product in base_products:
        test_products.extend(create_variations(base_product))

    # Generate noise products
    noise_products = [
        "Bàn học sinh", "Máy in văn phòng", "Nồi cơm điện",
        "Bếp từ", "Máy lọc nước", "Laptop gaming",
        "Điện thoại thông minh", "Tai nghe bluetooth"
    ]

    test_products.extend(noise_products)
    test_products = list(set(test_products))[:50]

    # Create and manage embeddings
    search_engine = DocumentEncoder()

    # Generate embeddings
    search_engine.generate_embeddings(test_products)

    # Load data
    search_engine.load_data(test_products)

    # Calculate Precision
    precision_1 = search_engine.calculate_precision(top_k=1)
    precision_3 = search_engine.calculate_precision(top_k=3)

    # Print results
    print("Precision@1:", precision_1['Precision@1'])
    print("Precision@3:", precision_3['Precision@3'])

    # Example search
    print("\nExample Search:")
    query = "Áo thun nam"
    results = search_engine.search(query, top_k=5, expected_top=3)
    print(f"Search results for '{query}':")
    for result in results:
        print(result)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error details: {e}")
        traceback.print_exc()