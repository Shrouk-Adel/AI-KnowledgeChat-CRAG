# AI-KnowledgeChat-CRAG
Corrective Retrieval-Augmented Generation Chatbot

## Overview

AI-KnowledgeChat-RAG is an AI-powered chatbot designed to enhance information retrieval and response accuracy. It primarily fetches knowledge from AI research topics, including **AI agents, prompt engineering, and adversarial attacks on LLMs**. If relevant information is not found in its internal knowledge base, it performs a **web search** to provide the best possible answer.

## 🌟 Features

AdaptiveRAG is an intelligent RAG system built with LangGraph that goes beyond traditional RAG implementations:

- **🧐 Document Relevance Grading**: Evaluates document relevance to filter out irrelevant content
- **✏️ Intelligent Query Rewriting**: Transforms user questions to optimize for better search results
- **🔎 Dynamic Web Search**: Falls back to web search when local knowledge is insufficient
- **🧩 Modular LangGraph Architecture**: Easily extensible workflow with clear separation of components
- **🚀 FastAPI Integration**: Production-ready API for easy deployment

🏗️ Architecture

![Screenshot 2025-02-25 132050](https://github.com/user-attachments/assets/679465e3-03ca-49e5-bba3-d5f0a3fa6960)

The system implements a sophisticated workflow:
1. **Retrieval**: Fetches potentially relevant documents from the vector store
2. **Relevance Grading**: Evaluates if documents truly answer the question
3. **Conditional Branching**: Takes action based on document relevance
   - If documents are relevant: Proceed to answer generation
   - If documents are irrelevant: Rewrite query and search the web
4. **Answer Generation**: Produces comprehensive answers using all relevant information

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- API keys for LLM services (Groq)
- API key for Tavily (web search)
## Technologies Used

- **Python** – Main programming language
- **FastAPI** – Web framework for API deployment
- **LangChain** – Used for retrieval, prompt management, and query refinement
- **ChromaDB** – Vector database for document storage and retrieval
- **OllamaEmbeddings** – Embedding model for vectorization
- **Tavily API** – Web search integration for external data retrieval
- **Groq LLM (Qwen-2.5-32B)** – Language model for generating responses

## Data Sources

The chatbot retrieves AI research content from the following sources:

- [Agent Mechanisms](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Prompt Engineering Guide](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/Shrouk-Adel/AI-KnowledgeChat-RAG.git
cd AI-KnowledgeChat-RAG
```

### 2. Create a Virtual Environment

```sh
python -m venv venv
# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file and add the required API keys:

```env
TAVILY_API_KEY=your_tavily_api_key
LangChain_API_Key=your_langchain_api_key
Groq_API_Key=your_groq_api_key
```

## Running the Application

Start the FastAPI server:

```bash
python App.py
```

Access the API at: `http://localhost:8000/query`

## API Usage

Send a **POST request** to `/query` with a JSON payload:

```json
{
  "question": "What are adversarial attacks on LLMs?"
}
```

**Example Response:**

```json
{
  "answer": "Adversarial attacks on LLMs involve modifying input prompts to exploit weaknesses in language models..."
}
```

## Future Improvements

- Enhance document ranking with more advanced relevance scoring.
- Expand the dataset to include more AI-related content.
- Implement caching mechanisms to reduce web search dependency.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for core RAG components
- [LangGraph](https://github.com/langchain-ai/langgraph) for the workflow framework
- [Groq](https://groq.com) for fast LLM inference
- [Tavily](https://tavily.com) for web search capabilities
- [Chroma](https://www.trychroma.com/) for vector storage

