from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "rag-documents"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled"
    )
    print(f"Created index: {index_name}")
else:
    print(f"Index '{index_name}' already exists")