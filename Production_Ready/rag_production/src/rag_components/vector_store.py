from langchain_community.vectorstores import FAISS

def VectorStore(split_docs,embeddings):
    print("vector_store")
    vector_db = FAISS.from_documents(split_docs,embeddings)
    retriever = vector_db.as_retriever(search_type="similarity",search_kwarg={"k":3})
    
    return vector_db , retriever