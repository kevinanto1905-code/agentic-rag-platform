import os
from unittest.mock import MagicMock

import openai
import pinecone

# Tests must never call the real OpenAI or Pinecone APIs.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
openai.OpenAI = MagicMock()
pinecone.Pinecone = MagicMock()