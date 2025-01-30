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
        print(f"Scraping error: {e}")
        return None

def process_content(content):
    # Clean and structure content
    processed = content
    return processed

def initialize_chatbot():
    # Scrape website content
    website_content = scrape_website("https://botpenguin.com/")
    processed_content = process_content(website_content)
    
    return processed_content

def chat_with_gpt(context, user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Use this knowledge base: {context}"},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    context = initialize_chatbot()
    
    if not context:
        print("Failed to initialize chatbot")
        return
    
    print("Chatbot initialized. Ask about BotPenguin!")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = chat_with_gpt(context, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()