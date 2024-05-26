from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
import boto3
from langchain.prompts import PromptTemplate
from QAsystem.ingestion import get_vector_store, data_ingestion
from langchain_community.embeddings import BedrockEmbeddings

# Initialize Bedrock client and embeddings
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Define the prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer and easy to understand wany to the question at the end but use at least 250 words with detailed explanations.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Function to get LLaMA2 LLM
def get_llm():
    print("Initializing LLM...")
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock)
    return llm

# Function to get response from LLM
def get_response_llm(llm, vectorstore_faiss, query):
    print("Setting up RetrievalQA...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print(f"Querying: {query}")
    answer = qa.invoke({"query": query})
    return answer["result"]

if __name__ == '__main__':
    # Uncomment the following lines if you need to ingest data and create vector store
    # print("Ingesting data and creating vector store...")
    # docs = data_ingestion()
    # vectorstore_faiss = get_vector_store(docs)

    print("Loading FAISS index...")
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    query = "What is RAG token?"
    llm = get_llm()
    
    print("Generating response...")
    response = get_response_llm(llm, faiss_index, query)
    print("Response:")
    print(response)
