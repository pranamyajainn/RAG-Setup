import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms.llm  import LLM
import requests


# Load environment variables from .env file
load_dotenv()

# Set up Groq LLM client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Custom LLM Class for Groq
class GroqLLM(LLM):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

    def _complete(self, prompt: str, **kwargs) -> str:
        # Synchronous completion method using the Groq API
        return self.groq_api_request(prompt)

    def groq_api_request(self, prompt: str) -> str:
        try:
            response = requests.post(
                url="https://api.groq.com/v1/generate",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"prompt": prompt, "model": self.model_name}
            )
            response.raise_for_status()
            return response.json().get("text", "No response")
        except requests.exceptions.RequestException as e:
            return f"Error with Groq API request: {e}"

# Initialize custom LLM
groq_llm = GroqLLM(model_name=GROQ_MODEL, api_key=GROQ_API_KEY)

# Create LLMPredictor with the custom LLM
llm_predictor = LLM(llm=groq_llm)

# Initialize the embedding model
embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')

# System configuration
PERSIST_DIR = "./storage"

# Initialize the index
def initialize_index(documents_path):
    print(f"Initializing index with documents from: {documents_path}")
    try:
        documents = SimpleDirectoryReader(
            documents_path,
            required_exts=['.txt', '.pdf', '.csv', '.xlsx', '.json']
        ).load_data()

        if not documents:
            raise ValueError(f"No valid documents found in {documents_path}.")

        print(f"Successfully loaded {len(documents)} documents.")

        # Create the index using LLMPredictor and embedding model
        index = VectorStoreIndex.from_documents(
            documents,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            show_progress=True
        )
        # Persist the index to disk
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    except Exception as e:
        print(f"Error initializing index: {e}")
        raise
    return index

# Query the index
def query_index(query, documents_path):
    print(f"Querying index with query: {query}")

    if not os.path.exists(PERSIST_DIR):
        # Initialize index if it doesn't exist
        index = initialize_index(documents_path)
    else:
        try:
            # Load the index from storage
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(
                storage_context,
                llm_predictor=llm_predictor,
                embed_model=embed_model
            )
        except Exception as e:
            print(f"Error loading index from storage: {e}")
            # Re-initialize index if loading fails
            index = initialize_index(documents_path)

    # Create a QueryEngine from the index
    query_engine = index.as_query_engine(llm_predictor=llm_predictor)

    # Query the index using the QueryEngine
    try:
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        print(f"Error querying index: {e}")
        raise

# Main function to test the code
if __name__ == "__main__":
    documents_path = "/Users/pranamyajain/PycharmProjects/FullStackReportGeneratingAgent/documents"  # Replace with the actual path
    query = "What is the meaning of life?"  # Example query

    try:
        result = query_index(query, documents_path)
        print(f"Query Result: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
