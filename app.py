import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = openai.OpenAI()

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

def chat_with_gpt(context, user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Use this knowledge base: {context}"},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

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
            st.session_state["context"] = context
            st.success("Content scraped successfully!")

if "context" in st.session_state:
    st.text_area("Scraped Content", st.session_state["context"], height=200)
    user_input = st.text_input("You:")
    if st.button("Send") and user_input:
        response = chat_with_gpt(st.session_state["context"], user_input)
        st.text_area("Bot:", response, height=100)
