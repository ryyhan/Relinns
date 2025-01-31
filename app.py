import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter  # LangChain chunker

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Chroma client with persistent storage
client_chroma = chromadb.PersistentClient(path="./chroma_db")

# Check if the collection exists and create it if not
def get_or_create_collection(collection_name):
    try:
        collection = client_chroma.get_collection(collection_name)
    except chromadb.errors.InvalidCollectionException:
        collection = client_chroma.create_collection(collection_name)
    return collection

# Create or get the collection to store embeddings
collection = get_or_create_collection("web_data_embeddings")

# Function to log scraped URLs
def log_scraped_url(url):
    with open("log.txt", "a") as f:
        f.write(url + "\n")

# Function to check if a URL has already been scraped
def is_url_scraped(url):
    if not os.path.exists("log.txt"):
        return False
    with open("log.txt", "r") as f:
        scraped_urls = f.read().splitlines()
    return url in scraped_urls

# Function to scrape a website
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
            
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
        response = client_openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        return f"Error generating embeddings: {e}"

# Function to store embeddings in ChromaDB with unique IDs
def store_embeddings(url, chunks, embeddings):
    try:
        valid_data = [(chunk, embedding) for chunk, embedding in zip(chunks, embeddings) if embedding is not None]
        if not valid_data:
            raise ValueError("No valid embeddings to store!")

        for i, (chunk, embedding) in enumerate(valid_data):
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": url}],  # Store URL as metadata
                ids=[f"{url}_chunk_{i}"]  # Unique ID per website
            )
        
        print(f"Total documents after insertion: {collection.count()}")
    except Exception as e:
        return f"Error storing embeddings: {e}"

# Function to search ChromaDB for similar documents
def search_similar_documents(query, top_k=5):
    try:
        query_embedding = generate_embeddings(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        print("ChromaDB Query Results:", results)
        
        if 'documents' in results and results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                print(f"Document from: {meta['source']} -> {doc[:200]}")
            return results['documents'][0]
        
        return ["No relevant documents found."]
    except Exception as e:
        return [f"Error searching ChromaDB: {e}"]

# Function to chat with GPT using the retrieved context
def chat_with_gpt(context, user_input):
    try:
        print(f"Using context: {context[:500]}")
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question concisely. If you don't know, say 'I don't know'.\n\nContext:\n{context}"},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with the chatbot: {str(e)}"

# Streamlit UI
st.title("Website Chatbot")
st.write("Enter a website URL to scrape and chat based on its content.")

if collection.count() > 0:
    st.success("Embeddings already exist in the database. You can start chatting!")
else:
    st.warning("No embeddings found. Please scrape a website first.")

# Chatbot Section
st.header("Chatbot")
user_input = st.text_input("You:")
if st.button("Send") and user_input:
    with st.spinner("Generating response..."):
        relevant_docs = search_similar_documents(user_input)
        context_for_gpt = "\n".join([doc if isinstance(doc, str) else str(doc) for doc in relevant_docs]) if relevant_docs else ""
        response = chat_with_gpt(context_for_gpt, user_input)
        st.write("Bot:", response)

st.divider()

# Scraping Section
st.header("Scrape and Embed Website Data")
st.write("Enter a website URL to scrape and generate embeddings.")

url = st.text_input("Website URL", "https://botpenguin.com/", key="scrape_url")

if st.button("Scrape Website"):
    if is_url_scraped(url):
        st.warning("This website has already been scraped. No need to scrape it again.")
    else:
        with st.spinner("Scraping content..."):
            context = scrape_website(url)
            if "Scraping error" in context:
                st.error(context)
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_text(context)
                embeddings = [generate_embeddings(chunk) for chunk in chunks]

                if all(isinstance(embedding, list) for embedding in embeddings):
                    store_embeddings(url, chunks, embeddings)  # Pass URL
                    log_scraped_url(url)  
                    st.success(f"Content from {url} stored successfully!")
                else:
                    st.error("Error generating embeddings for one or more chunks.")
