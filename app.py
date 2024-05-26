import json
import os 
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from QAsystem.ingestion import data_ingestion, get_vector_store
from QAsystem.retrievalandgeneration import get_llm, get_response_llm

# Initialize Bedrock client and embeddings
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Main function for the Streamlit app
def main():
    # Set up Streamlit page configuration
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using langchain and AWS Bedrock")
    
    # Input field for user to ask a question
    user_question = st.text_input("Ask a question from the PDF files")
    
    # Sidebar for vector store updates and running the QA system
    with st.sidebar:
        st.title("Update or Create the Vector Store")
        
        # Button to update the vector store
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                # Ingest data and create vector store
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
                
        # Button to run the QA system
        if st.button("Run"):
            with st.spinner("Processing..."):
                # Load the FAISS index and initialize the LLM
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llm()
                
                # Get the response from the LLM and display it
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

# Entry point for the script
if __name__ == "__main__":
    main()
