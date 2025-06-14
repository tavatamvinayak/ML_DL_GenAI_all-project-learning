
from langchain_openai import OpenAIEmbeddings
import os
# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

def Embeddings():
    print("Embeddings")
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    return embeddings