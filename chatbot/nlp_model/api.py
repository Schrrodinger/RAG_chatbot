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

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your RAG components
from embeddings import DocumentEncoder
from retriever import ProductRetriever
from generator import ResponseGenerator
from rag_pipeline import RAGPipeline

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
generator = ResponseGenerator()
rag_pipeline = RAGPipeline(retriever, generator)


import json
from pathlib import Path
import pandas as pd

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
# Load products, QnAs, and reviews
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
from preprocessing import merge_datasets

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
    try:
        logger.info(f"Received chat request: {request.message}")

        # Process through RAG pipeline
        result = rag_pipeline.process_query(
            query=request.message,
            history=[msg.dict() for msg in request.history]
        )

        # Sanitize the response to ensure JSON compliance
        sanitized_result = sanitize_for_json(result)

        response = ChatResponse(
            content=sanitized_result["response"],
            products=sanitized_result.get("relevant_products", [])[:3]  # Top 3 products
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
