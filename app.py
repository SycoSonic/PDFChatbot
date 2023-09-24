# app.py
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings  # Importing OpenAIEmbeddings

# Load .env file
load_dotenv()

# Access .env variables
openai_key = os.getenv('OPENAI_KEY')
model_name = os.getenv('MODEL_NAME')
embedding_model_name = os.getenv('EMBEDDING_MODEL')  # Accessing EMBEDDING_MODEL from .env

# Initialize OpenAI LLM and Embeddings from LangChain
llm = OpenAI(openai_api_key=openai_key, model_name=model_name)  # Using MODEL_NAME from .env
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key, model_name=embedding_model_name)  # Initializing Embeddings Model with API key and EMBEDDING_MODEL from .env

def main():
    st.title('ChatGPT-powered PDF Chatbot')
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            # Create a file object from the uploaded file
            pdf_file = BytesIO(uploaded_file.read())
            
            # Create a PDF reader object using the corrected class name
            pdf_reader = PdfReader(pdf_file)
            
            # Extract text from each page
            pdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()
            
            # Display extracted text
            st.text_area("Extracted Text", pdf_text)
            
            # User input for queries
            user_query = st.text_input("Ask a question about the uploaded document:")
            
            if user_query and pdf_text.strip():
                # Embed the documents and the query
                embedded_documents = embeddings_model.embed_documents([pdf_text])
                embedded_query = embeddings_model.embed_query(user_query)
                
                # Here you can perform semantic search or other operations using the embeddings
                
                # Create a prompt template
                prompt_template = PromptTemplate.from_template("{document_text}\n{user_query}")
                
                # Format the prompt with extracted PDF text and user query
                formatted_prompt = prompt_template.format(document_text=pdf_text, user_query=user_query)
                
                # Generate response from LLM based on the formatted prompt
                response = llm(formatted_prompt)
                st.text_area("Response", response)
            
            st.success("File uploaded and processed successfully!")
        except Exception as e:
            # Display an error message if file processing fails
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()





