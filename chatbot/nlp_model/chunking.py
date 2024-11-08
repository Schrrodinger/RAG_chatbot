from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Step 1: Chunking Large Texts
def chunk_text(text, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)

    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks


# Step 2: DocumentEncoder for Embedding Each Chunk
class DocumentEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, chunks):
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        return embeddings


# Step 3: VectorSearch with FAISS
class VectorSearch:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices


# Step 4: Main Function to Run the Workflow
def test_rag_system():
    # Sample product descriptions
    product_descriptions = [
        "This is a great smartphone with a powerful battery and sharp camera.",
        "The laptop has excellent performance and a durable build, perfect for professional use."
    ]

    # Step 1: Chunk Texts
    all_chunks = []
    for description in product_descriptions:
        chunks = chunk_text(description)
        all_chunks.extend(chunks)

    # Step 2: Encode Chunks
    encoder = DocumentEncoder()
    embeddings = encoder.encode_chunks(all_chunks)
    embedding_array = embeddings.cpu().numpy()  # Convert to NumPy for FAISS

    # Step 3: Initialize Vector Search and Add Embeddings
    dimension = embeddings.shape[1]
    search_index = VectorSearch(dimension)
    search_index.add_embeddings(embedding_array)

    # Step 4: Process Query and Find Relevant Chunks
    query = "Find a product with a good battery life."
    query_embedding = encoder.encode_chunks([query]).cpu().numpy()
    distances, indices = search_index.search(query_embedding, top_k=5)

    # Display Results
    relevant_chunks = [all_chunks[idx] for idx in indices[0]]
    print("Relevant Chunks:")
    for chunk in relevant_chunks:
        print(chunk)


# Run the test function
if __name__ == "__main__":
    test_rag_system()
