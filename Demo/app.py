


import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os


# Function to load and process documents
@st.cache_resource
def load_documents():
    # For demo, create a sample product manual
    sample_manual = """
    Product: SmartHome Device
    Version: 2.1
    Features:
    - Voice-activated controls
    - Wi-Fi connectivity (2.4GHz and 5GHz)
    - Compatible with iOS and Android
    Troubleshooting:
    - If device doesn't connect to Wi-Fi, reset by holding power button for 10 seconds
    - For voice recognition issues, ensure microphone is not obstructed
    Setup Instructions:
    1. Plug in the device
    2. Download SmartHome app
    3. Follow in-app setup wizard
    """
    
    # Save sample manual to temporary file
    with open("manual.txt", "w") as f:
        f.write(sample_manual)
    
    # Load and split document
    loader = TextLoader("manual.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)



# Function to create RAG pipeline
@st.cache_resource
def create_rag_pipeline():
    # Load documents
    documents = load_documents()
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Initialize language model
    llm = HuggingFacePipeline.from_model_id(
        model_id="distilgpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100}
    )
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain


# Streamlit page configuration
st.set_page_config(page_title="Product Manual Q&A", layout="wide")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    
# Main Streamlit app
def main():
    st.title("SmartHome Device Support Chatbot")
    st.write("Ask any question about your SmartHome Device!")

    # Load RAG pipeline
    qa_chain = create_rag_pipeline()

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What's your question about the SmartHome Device?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from RAG pipeline
        with st.spinner("Thinking..."):
            response = qa_chain({"query": prompt})
            answer = response["result"]
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

        # Display source documents
        with st.expander("Source Documents"):
            for doc in response["source_documents"]:
                st.write(doc.page_content)
                
                
                
if __name__ == "__main__":
    main()