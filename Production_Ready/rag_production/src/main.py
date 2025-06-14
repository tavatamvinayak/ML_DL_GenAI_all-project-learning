from rag_components.documents_loaders import load_documents
from rag_components.text_splitter import TextSplitter
from rag_components.embeddings import Embeddings
from rag_components.vector_store import VectorStore
from rag_components.models import Models

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough , RunnableLambda
from langchain_core.output_parsers import StrOutputParser



from fastapi import FastAPI,UploadFile,File,HTTPException
import uvicorn
from pydantic import BaseModel

import os 
from dotenv import load_dotenv

# load .env environment variables
load_dotenv()

# RAG Components
documents = load_documents('./documents')
split_docs = TextSplitter(documents)
embeddings = Embeddings()
vector_db , retriever = VectorStore(split_docs,embeddings)
llm =Models()

prompt = PromptTemplate(
    template=""" you are a helpful assistant.
    answer only from the provided transcript context.
    if the context is insufficient, just say you don't know.
    Context:{context}
    
    Question: {question}
    """,
    input_variables=['context','question']
)
#Format retrieved docs into a string
def format_docs(retrieved_docs):
    return '\n\n'.join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    'context':retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# result = main_chain.invoke("who is vinayak?")
# print("Answer :- ",result)

from pydantic import BaseModel

class QueryParams (BaseModel):
    query:str

app = FastAPI(title="RAG FastApi")

@app.post('/query/')
async def query_rag(params:QueryParams):
    try:
        query_text = params.query
        print(f"Query: {query_text}")
        try:
            response = main_chain.invoke(query_text)
            return {'response': response}
        except Exception as e:
            raise HTTPException(status_code=300,detail="Model Error to Answer")
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


def main():
    print("Hello from rag-production!")
    uvicorn.run(app ,host="0.0.0.0",port=8080)


if __name__ == "__main__":
    main()
