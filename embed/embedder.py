# === Core ===
import os
import sys
import time
import random

# === Data Handling ===
import pandas as pd
import json

# === NLP + Progress ===
from tqdm import tqdm
import tiktoken

# === OpenAI Client ===
import openai

# === Typing (optional but useful) ===
from typing import List, Dict, Tuple, Optional

from config import client, EMBEDDING_MODEL

def get_embedding(text: str) -> list[float]:
    if not text.strip():
        return []
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding