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
import logging

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

# logging
logging.basicConfig(level=logging.INFO)

# Check if the collection_name is in the list of existing collections
try:
    chroma_client.get_collection(collection_name)
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

def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=400):
    chunks = []
    current_chunk = ""
    overlap_text = ""
    sentences = text.split('. ')  # Simple sentence splitting based on periods
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = overlap_text
        current_chunk += sentence + ". "
        overlap_text = " ".join(current_chunk.split()[-chunk_overlap:])
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def main():
    st.title('ChatGPT-powered PDF Chatbot')
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            pdf_file = BytesIO(uploaded_file.read())
            pdf_text = extract_text_from_pdf(pdf_file)
            
            # Use the refined text splitting logic
            documents_str_list = split_text_into_chunks(pdf_text)
            
            logging.info(f"Number of chunks created: {len(documents_str_list)}")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Create embeddings for each chunk and add them to the collection
            for i, chunk in enumerate(documents_str_list):
                # Update the progress bar
                progress_bar.progress((i + 1) / len(documents_str_list))
                document_id = f"{uuid.uuid4()}_chunk_{i}"
                metadata = {"title": uploaded_file.name, "upload_timestamp": datetime.utcnow().isoformat(), "chunk_id": document_id}
                
                # Log the length of each chunk
                logging.info(f"Length of Chunk {i}: {len(chunk)}")
                
                # Display progress message on Streamlit GUI
                st.text(f"Creating embedding for chunk {i}...")
                
                embedding = text_embedding(chunk)
                
                # Log the size and a sample of the embedding
                logging.info(f"Embedding for chunk {i}: Size: {len(embedding)}, Sample: {embedding[:5]}")
                
                # Get the collection object
                collection = chroma_client.get_collection(collection_name)
                
                # Add the embedding to the collection, ensuring that only serializable data is added
                collection.add(ids=[document_id], embeddings=[embedding], metadatas=[metadata])
            
            st.text_area("Extracted Text", pdf_text)
            
            user_query = st.text_input("Ask a question about the uploaded document:")
            
            if user_query and pdf_text.strip():
                # Adjust the n_results parameter
                n_results = st.slider('Number of Results to Retrieve', min_value=1, max_value=10, value=5)
                
                query_embedding = text_embedding(user_query)
                results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
                
                # Retrieve the actual text content of the chunk/document
                aggregated_text = ""
                for result in results:
                    document_id = result['id']
                    metadata = result['metadata']
                    
                    # Retrieve the actual text content using metadata or other means
                    chunk_id = metadata.get('chunk_id')
                    chunk_index = int(chunk_id.split('_')[-1])  # Extracting index from chunk_id
                    chunk_text = documents_str_list[chunk_index]  # Getting the actual chunk text using the index
                    
                    aggregated_text += chunk_text + "\n"
                
                # Implement additional logic to prioritize or filter the retrieved chunks/documents
                sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
                
                # Aggregate the sorted results
                aggregated_text = ""
                for result in sorted_results:
                    document_id = result['id']
                    metadata = result['metadata']
                    chunk_id = metadata.get('chunk_id')
                    chunk_index = int(chunk_id.split('_')[-1])
                    chunk_text = documents_str_list[chunk_index]
                    aggregated_text += chunk_text + "\n"
                
                # Generate the response based on the aggregated text
                prompt_template = PromptTemplate.from_template("{document_text}\n{user_query}")
                formatted_prompt = prompt_template.format(document_text=aggregated_text, user_query=user_query)
                response = llm(formatted_prompt)
                st.text_area("Response", response)
            
            st.success("File uploaded and processed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
if __name__ == "__main__":
    main()
