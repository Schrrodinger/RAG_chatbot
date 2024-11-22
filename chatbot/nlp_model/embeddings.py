import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

class DocumentEncoder:
    def __init__(self, model_name="Qwen/Qwen2-1.5B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 4  # Reduce batch size to manage memory

        # Quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model and tokenizer
        try:
            print(f"Attempting to load model on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                use_auth_token=True, 
                quantization_config=quantization_config
            )
            print(f"Model loaded successfully on {self.device}.")
        except RuntimeError as e:
            print(f"CUDA Error: {e}. Falling back to CPU...")
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_auth_token=True).to(self.device)
            print("Model loaded successfully on CPU.")

    def encode_documents(self, documents):
        """
        Encode documents into embeddings using mean pooling over hidden states.
        """
        embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            inputs = self.tokenizer(batch_docs, return_tensors="pt", truncation=True, padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]
            masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            mean_pooled = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

            embeddings.append(mean_pooled.cpu().numpy())  # Move back to CPU for storage
        return np.vstack(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into an embedding using mean pooling.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        mean_pooled = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        return mean_pooled.cpu().numpy()
