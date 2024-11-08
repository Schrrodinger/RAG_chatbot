from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import json
from pathlib import Path
import sys
import os

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


def load_products():
    try:
        products_file = Path(__file__).parent / "sample_products.json"
        if products_file.exists():
            with open(products_file, "r", encoding="utf-8") as f:
                products = json.load(f)
                logger.info(f"Loaded {len(products)} products successfully")
                return products
        else:
            logger.warning(f"sample_products.json not found at {products_file}")
            return []
    except Exception as e:
        logger.error(f"Error loading products: {str(e)}")
        return []


# Load and index products on startup
@app.on_event("startup")
async def startup_event():
    try:
        products = load_products()
        if products:
            retriever.index_products(products)
            logger.info(f"Indexed {len(products)} products successfully")
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

        response = ChatResponse(
            content=result["response"],
            products=result.get("relevant_products", [])[:3]  # Top 3 products
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
