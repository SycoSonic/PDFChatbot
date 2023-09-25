# app.py
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import chromadb
import openai  # Import the OpenAI library
import uuid
from datetime import datetime

# Load .env file
load_dotenv()

# Access .env variables
openai_key = os.getenv('OPENAI_KEY')
model_name = os.getenv('MODEL_NAME')
embedding_model_name = os.getenv('EMBEDDING_MODEL')

# Initialize OpenAI instance
llm = OpenAI(openai_api_key=openai_key, model_name=model_name)

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# Define the collection name
collection_name = "pdf_embeddings"

# Get the list of existing collections
existing_collections = chroma_client.list_collections()

# Check if the collection_name is in the list of existing collections
try:
    documents = chroma_client.get_collection(collection_name).get(include=["metadatas", "documents"])
except Exception as e:
    chroma_client.create_collection(collection_name)

def text_embedding(text):
    response = openai.Embedding.create(model=embedding_model_name, input=text)
    return response["data"][0]["embedding"]

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def main():
    st.title('ChatGPT-powered PDF Chatbot')
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            pdf_file = BytesIO(uploaded_file.read())
            pdf_text = extract_text_from_pdf(pdf_file)
            st.text_area("Extracted Text", pdf_text)
            document_id = str(uuid.uuid4())
            metadata = {"title": uploaded_file.name, "upload_timestamp": datetime.utcnow().isoformat()}
            
            # Create the embedding using the text_embedding function
            embedding = text_embedding(pdf_text)
            
            # Get the collection object
            collection = chroma_client.get_collection(collection_name)
            
            # Add the embedding to the collection
            collection.add(ids=[document_id], embeddings=[embedding], metadatas=[metadata])
            
            user_query = st.text_input("Ask a question about the uploaded document:")
            
            if user_query and pdf_text.strip():
                query_embedding = text_embedding(user_query)
                results = collection.query(query_embeddings=[query_embedding], n_results=5)
                
                prompt_template = PromptTemplate.from_template("{document_text}\n{user_query}")
                formatted_prompt = prompt_template.format(document_text=pdf_text, user_query=user_query)
                response = llm(formatted_prompt)
                st.text_area("Response", response)
            
            st.success("File uploaded and processed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

