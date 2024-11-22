import math
import re

def sanitize_for_json(data):
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0
    return data

def preprocess_for_qwen(query):
    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", "", query)
    return query
