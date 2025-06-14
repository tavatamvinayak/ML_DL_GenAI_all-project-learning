from langchain_openai import ChatOpenAI
import os
# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()
def Models ():
    print("models")
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o",temperature=0.5)
    return llm