from langchain.text_splitter import RecursiveCharacterTextSplitter
import os



def TextSplitter(documents):
    print("TextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs