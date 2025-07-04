import os
from dotenv import load_dotenv
from openai import OpenAI  # Correct usage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Prefer from .env

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"  


VECTOR_DB_TYPE = "chroma"  
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 10

client = OpenAI(api_key=OPENAI_API_KEY)