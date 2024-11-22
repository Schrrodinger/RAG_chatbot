from typing import List, Dict
import nltk
import spacy

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

import pandas as pd
import re
import string

def merge_datasets(products, qnas=None, reviews=None):
    # Convert lists into DataFrames
    products_df = pd.DataFrame(products)

    # No need to merge, return products data only
    merged_data = products_df.to_dict(orient='records')
    return merged_data

def preprocess_text(text: str) -> str:
    """Perform text preprocessing such as lowering case, removing punctuation, etc."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_product_documents(products: List[Dict]) -> List[str]:
    """Convert product data to searchable documents."""
    documents = []
    for product in products:
        # Preprocess the text fields before combining
        name = preprocess_text(product.get('name', ''))
        description = preprocess_text(product.get('description', ''))
        specifications = preprocess_text(product.get('specifications', ''))

        # Combine all the fields into a single document
        doc = f"Name: {name}\nDescription: {description}\nSpecifications: {specifications}"
        documents.append(doc)
    return documents