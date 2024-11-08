import uvicorn
import logging
from pathlib import Path
import json
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to localhost
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)]
    )