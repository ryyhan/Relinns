import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Chroma client
client_chroma = chromadb.Client()

# Check if the collection exists and create it if not
def get_or_create_collection(collection_name):
    try:
        # Try to get an existing collection
        collection = client_chroma.get_collection(collection_name)
    except ValueError:
        # If the collection doesn't exist, create a new one
        collection = client_chroma.create_collection(collection_name)
    return collection

# Create or get the collection to store embeddings
collection = get_or_create_collection("web_data_embeddings")

# Function to scrape a website
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unnecessary elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
            
        # Extract main content
        content = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'p']:
            for element in soup.find_all(tag):
                content.append(element.get_text(strip=True))
                
        return ' '.join(content)
    
    except Exception as e:
        return f"Scraping error: {e}"

# Function to generate embeddings for the scraped content
def generate_embeddings(text):
    try:
        # Using the new API method for generating embeddings
        response = client_openai.embeddings.create(
            model="text-embedding-3-small",  # Specify the correct model
            input=text
        )
        embeddings = response.data[0].embedding  # Correct way to access the embeddings
        return embeddings
    except Exception as e:
        return f"Error generating embeddings: {e}"

# Function to store embeddings in ChromaDB
def store_embeddings(chunks, embeddings):
    try:
        # Store each chunk and its embedding
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": "web_scraped_data"}],
                ids=[f"chunk_{i}"]
            )
    except Exception as e:
        return f"Error storing embeddings: {e}"

# Function to search ChromaDB for similar documents
def search_similar_documents(query, top_k=5):
    try:
        query_embedding = generate_embeddings(query)
        
        # Search ChromaDB for the top-k most relevant documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Flatten the results to get a list of strings
        documents = results['documents'][0]  # Access the first list of documents
        return documents
    except Exception as e:
        return f"Error searching ChromaDB: {e}"

# Function to chat with GPT using the context from scraped content
def chat_with_gpt(context, user_input):
    try:
        # Use the new method for completions with a conversational prompt
        response = client_openai.chat.completions.create(
            model="gpt-4-turbo-preview",  # Replace with the correct model name
            messages=[
                {"role": "system", "content": f"Context:\n{context}"},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150  # Adjust token limit as needed
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with the chatbot: {str(e)}"

# Streamlit UI
st.title("Website Chatbot")
st.write("Enter a website URL to scrape and chat based on its content.")

url = st.text_input("Website URL", "https://botpenguin.com/")

if st.button("Scrape Website"):
    with st.spinner("Scraping content..."):
        context = scrape_website(url)
        if "Scraping error" in context:
            st.error(context)
        else:
            # Use LangChain's RecursiveCharacterTextSplitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Number of characters per chunk
                chunk_overlap=50,  # Overlap between chunks
                length_function=len,  # Function to measure chunk size
                separators=["\n\n", "\n", " ", ""]  # Split by paragraphs, lines, and words
            )
            
            # Split the scraped content into chunks
            chunks = text_splitter.split_text(context)
            
            # Generate embeddings for each chunk
            embeddings = [generate_embeddings(chunk) for chunk in chunks]
            
            # Store embeddings in ChromaDB
            if all(isinstance(embedding, list) for embedding in embeddings):  # Check if all embeddings are valid
                store_embeddings(chunks, embeddings)
                st.session_state["context"] = context
                st.session_state["chunks"] = chunks
                st.success("Content scraped, chunked, and embeddings stored successfully!")
            else:
                st.error("Error generating embeddings for one or more chunks.")

if "context" in st.session_state:
    st.text_area("Scraped Content", st.session_state["context"], height=200)
    user_input = st.text_input("You:")
    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            relevant_docs = search_similar_documents(user_input)
            
            # Ensure relevant_docs is a list of strings
            context_for_gpt = "\n".join([doc if isinstance(doc, str) else str(doc) for doc in relevant_docs]) if relevant_docs else st.session_state["context"]
            
            # Get response from GPT based on the context
            response = chat_with_gpt(context_for_gpt, user_input)
            st.text_area("Bot:", response, height=100)