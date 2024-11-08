import numpy as np
from typing import List, Dict, Union
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    def __init__(self, model_name: str = "vinai/phobert-base"):
        """
        Initialize with PhoBERT base model for Vietnamese text encoding.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._use_fallback_model()

    def _use_fallback_model(self):
        """Use smaller model as fallback."""
        try:
            fallback_model = "vinai/phobert-base-v2"
            logger.info(f"Loading fallback model: {fallback_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModel.from_pretrained(fallback_model)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Fallback model loaded successfully")
        except Exception as e:
            logger.error(f"Critical: Fallback model failed: {str(e)}")
            raise RuntimeError("Unable to initialize any model")

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts."""
        try:
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Error in batch encoding: {str(e)}")
            return np.zeros((len(texts), self.model.config.hidden_size))

    def encode(self, texts: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def encode_documents(self, documents: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
        """Encode product documents."""
        logger.info("Encoding documents...")
        document_embeddings = {}

        for doc in documents:
            doc_id = str(doc.get('id', len(document_embeddings)))
            doc_embeddings = {}

            # Encode name
            if 'name' in doc and doc['name']:
                doc_embeddings['name'] = self.encode(doc['name'])[0]

            # Encode description
            if 'description' in doc and doc['description']:
                doc_embeddings['description'] = self.encode(doc['description'])[0]

            # Encode specifications
            if 'specifications' in doc and doc['specifications']:
                spec_text = ' '.join(f"{k}: {v}" for k, v in doc['specifications'].items())
                doc_embeddings['specifications'] = self.encode(spec_text)[0]

            if doc_embeddings:
                document_embeddings[doc_id] = doc_embeddings

        return document_embeddings

    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: Dict[str, Dict[str, np.ndarray]]) -> \
    Dict[str, float]:
        """Compute similarity between query and documents."""
        similarities = {}

        for doc_id, doc_embeddings in document_embeddings.items():
            field_similarities = []

            for field_embedding in doc_embeddings.values():
                similarity = np.dot(query_embedding, field_embedding)
                field_similarities.append(similarity)

            similarities[doc_id] = max(field_similarities) if field_similarities else 0.0

        return similarities

    def find_similar_documents(self, query: str, document_embeddings: Dict[str, Dict[str, np.ndarray]],
                               top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """Find most similar documents to query."""
        try:
            query_embedding = self.encode(query)[0]
            similarities = self.compute_similarity(query_embedding, document_embeddings)

            sorted_results = sorted(
                [{"id": k, "score": float(v)} for k, v in similarities.items()],
                key=lambda x: x["score"],
                reverse=True
            )

            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []

    def generate_response(self,query, relevant_products):
        if not relevant_products:
            return "Xin lỗi, chúng tôi không tìm thấy sản phẩm phù hợp với tìm kiếm của bạn"
        response = f"Dựa vào '{query}', đây là những sản phẩm phù hợp: \n"
        for product in relevant_products:
            response += f"{product['name']} - {product['description']}\n"

        return response