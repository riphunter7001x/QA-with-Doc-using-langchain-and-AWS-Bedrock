from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import json
import os
import sys
import boto3

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    print("Starting data ingestion...")
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")
    
    return docs

def get_vector_store(docs):
    print("Creating vector store...")
    vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    print("Vector store saved locally as 'faiss_index'.")
    return vector_store_faiss
    
if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)
    print("Process completed.")
