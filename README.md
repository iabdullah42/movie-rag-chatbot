# Movie Recommender & Q&A Chatbot (RAG with OpenAI + Pinecone)

**A chatbot that answers questions about movies and recommends films. It searches a vector database of Rotten Tomatoes movie data and uses an LLM to write the answer.**

## How it works

```
Rotten Tomatoes movie data ─► text chunks (≤1,536 tokens, sentence-aware)
                             ─► OpenAI embeddings ─► Pinecone vector index

User question ─► embedded ─► semantic search in Pinecone ─► best-matching movie text
                            ─► LangChain QA chain (OpenAI chat model) ─► answer
                            ─► Gradio web interface
```

This pattern is called **retrieval-augmented generation (RAG)**. The LLM answers from retrieved source text rather than from memory alone, which makes answers more grounded and lets the knowledge base be updated without retraining.

## What's in the notebook (`Movies2.ipynb`)

- **Chunking and tokenization:** splits movie descriptions into sentence-aligned chunks with `nltk` and `tiktoken` so each fits the embedding model's token limit
- **Embeddings and indexing:** creates OpenAI embeddings and upserts them into Pinecone in batches
- **Retrieval + generation:** `ask_a_question()` embeds the question, finds the closest match, and passes it to a LangChain QA chain
- **Sentiment layer (experimental):** VADER sentiment combined with Tomatometer / audience status to group films from negative to very positive
- **Interface:** a Gradio app ("Movie Recommendation System") where users type their thoughts about a movie

## Run it

1. `pip install pandas openai langchain langchain-openai pinecone-client tiktoken nltk vaderSentiment gradio python-dotenv`
2. Copy `.env.example` to `.env` and add your own `OPENAI_API_KEY` and `PINECONE_API_KEY`. `.env` is git-ignored, so keys are never committed.
3. Place the Rotten Tomatoes movies dataset at `Resources/rotten_tomatoes_movies.csv`.
4. Run `Movies2.ipynb`.

**Tools:** Python, OpenAI API, Pinecone, LangChain, tiktoken, NLTK, VADER, Gradio, pandas

*Built as a project for the OSU AI Bootcamp.*
