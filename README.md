# Website Chatbot with ChromaDB and OpenAI

This project is a **website chatbot** that scrapes webpage content, generates embeddings using OpenAI, stores them in **ChromaDB**, and allows users to query relevant information using a chatbot interface built with **Streamlit**.

## Features
- 🕵️ **Web Scraping**: Extracts text content from a given website.
- 🧠 **Embedding Generation**: Uses OpenAI's `text-embedding-3-small` model to create embeddings.
- 📦 **ChromaDB Integration**: Stores embeddings persistently.
- 🔍 **Similarity Search**: Retrieves relevant documents based on user queries.
- 🤖 **Chatbot Interface**: Uses GPT-4o-mini to answer queries based on stored content.

## Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/ryyhan/Relinns.git
   cd website-chatbot
   ```
2. **Create a virtual environment (optional, but recommended)**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables Create a .env file and add your OpenAI API key**

    
    ```sh
    OPENAI_API_KEY=your_openai_api_key
## Usage

Start the Streamlit App
```sh
streamlit run app.py
```