
from langchain.vectorstores import pinecone
import os

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='gcp-starter')

index_name="langchain-demo"

if index_name not in pinecone.lidt_indexes():
    pinecone.create_index(name=index_name, metric="cosine",dimension=768)
    docsearch=pinecone.from_documents(docs,embeddings,index_name=index_name)
else:
    docseach=pinecone.from_existing_index(index_name, embeddings)


from langchain.llms import HuggingFaceHub 

repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm=HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature":0.8,"top_k":50},
    huggingfacehub_api_tokens=os.getenv('HUGGINGFACE_API_KEY'))

from langchain.prompts import PromptTemplate

template="""You are a fortune teller. These Human will ask you a questions about their life. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer withifrom langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os


loader=TextLoader('./horoscope.txt')
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs=text_splitter.split_documents(documents)

embeddings=HuggingFaceEmbeddings()
n 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""
prompt=PromptTemplate(
template=template,
imput_variables=["context","question"]

)


from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

rag_chain=(

    {"context":docsearch.as_retriever(),"question":RunnablePassthrough()}


    |prompt
    |llm
    |StrOutputParser()
)
