# YouTube Videos Insights

An interactive web application that allows users to ask questions about YouTube videos. This app fetches video transcripts, splits them into manageable chunks, creates embeddings for semantic search, and answers user queries using **Hugging Face LLMs**.

-----

## Features

  - **Automatic Transcript Fetching**: Fetches transcripts from YouTube videos automatically using the `youtube-transcript-api` library.
  - **Efficient Retrieval**: Splits transcripts into manageable chunks using **LangChain** to optimize the retrieval process.
  - **Semantic Search**: Creates **vector embeddings** of transcript chunks using **Hugging Face embedding models** to enable context-aware search.
  - **Fast Performance**: Stores embeddings in **FAISS** (Facebook AI Similarity Search) for fast and efficient semantic similarity lookups.
  - **AI-Powered Answers**: Answers user questions using **Hugging Face LLMs** (`openai/gpt-oss-120b`), providing responses based solely on the video's content.
  - **User-Friendly Interface**: Built with **Streamlit** for a simple and interactive user experience.

-----

## Demo

1.  Enter the YouTube video ID in the input box on the sidebar.
2.  Type your question about the video's content.
3.  Receive an AI-generated answer based on the video's transcript.

-----

## Installation

To get started, follow these steps to set up the application on your local machine.

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/gauravpatil11/YT-video-Chatbot-using-Langchain
    cd YT-video-Chatbot-using-Langchain
    ```

2.  **Create a virtual environment** (optional but recommended):

    ```bash
    # Linux / macOS
    python -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set your Hugging Face API token**: This token is required to use the Hugging Face models. You can get one by signing up on the Hugging Face website.

    ```bash
    # Linux / macOS
    export HUGGINGFACEHUB_API_TOKEN="your_hf_token"

    # Windows
    set HUGGINGFACEHUB_API_TOKEN=your_hf_token
    ```

-----

## Usage

Once you've completed the installation, you can run the application.

1.  **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

2.  **Access the app**:

    Open the URL shown in your terminal (usually `http://localhost:8501`) to access the app in your web browser.
