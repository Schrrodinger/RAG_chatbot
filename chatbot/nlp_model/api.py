from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import json
from pathlib import Path
import sys
import os
import math
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your RAG components
from embeddings import DocumentEncoder
from retriever import ProductRetriever
from preprocessing import merge_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class ChatMessage(BaseModel):
    role: str
    parts: str

    def to_dict(self):
        return {
            "role": self.role,
            "parts": self.parts
        }


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

    class Config:
        arbitrary_types_allowed = True


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    products: Optional[List[Dict]] = None


# Initialize RAG components as global variables
encoder = DocumentEncoder()
retriever = ProductRetriever(encoder)

# Qwen2-1.5B integration for response generation
model_name = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True, use_auth_token=True)
generator_model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,  use_auth_token=True)


class QwenResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response_with_qwen(self, query, relevant_products, conversation_history):
        """
        Generate response using Qwen2-1.5B model by incorporating query, relevant products, and conversation history.
        """
        # Prepare input text
        history_text = "\n".join([f"User: {msg['parts']}" if msg["role"] == "user" else f"Assistant: {msg['parts']}" 
                                  for msg in conversation_history])
        product_info = "\n".join([f"Product: {prod['name']} - {prod['description']}" for prod in relevant_products])
        input_text = f"{history_text}\nRelevant Products:\n{product_info}\nUser: {query}\nAssistant:"

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )

        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()


generator = QwenResponseGenerator(generator_model, tokenizer)


# Data loading and sanitization
def sanitize_for_json(data):
    """Replace non-JSON-compliant values with suitable defaults."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0  # Replace NaN or Infinity with 0 (or any default value)
    return data


def load_products():
    datasets = {}
    try:
        data_folder = Path(__file__).parent
        # Load data from CSV file
        products_df = pd.read_csv(data_folder / "usage_data.csv", encoding="utf-8")

        # Convert to dictionary for use in downstream processes
        datasets['products'] = products_df.to_dict(orient='records')
        return datasets
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        return {'products': []}


# Load and index products on startup
@app.on_event("startup")
async def startup_event():
    try:
        datasets = load_products()
        products = datasets.get('products', [])

        # Merge the datasets (products only in this case)
        merged_data = merge_datasets(products)

        # Index merged products
        if merged_data:
            retriever.index_products(merged_data)
            logger.info(f"Indexed {len(merged_data)} products successfully")
        else:
            logger.error("No products found to index")
    except Exception as e:
        logger.error(f"Error indexing products: {str(e)}")
        raise


@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process user queries and generate a response using Qwen2-1.5B.
    """
    try:
        logger.info(f"Received chat request: {request.message}")

        # Retrieve relevant products
        relevant_products = retriever.retrieve(request.message)

        # Generate response using Qwen2-1.5B
        response_content = generator.generate_response_with_qwen(
            query=request.message,
            relevant_products=relevant_products[:3],  # Limit to top 3 products
            conversation_history=[msg.to_dict() for msg in request.history]
        )
        
        response = ChatResponse(
            content=response_content,
            products=relevant_products[:3]
        )

        logger.info("Generated response successfully")
        return response

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau."
        )


@app.options("/chat")
async def chat_options():
    return {}
