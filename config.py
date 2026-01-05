
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db_semantic_chunking_percentile_80")

# --- Models ---
# Embedding model for vector store
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Main LLM for generation (Ollama)
MODEL_ID = "qwen2.5:3b"

# Vision Language Model for Image Description
VLM_MODEL_ID = "qwen2.5vl:3b"

# Ollama API Config
OLLAMA_BASE_URL = "http://localhost:11434"

# Cross-Encoder for Reranking
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Ingestion Config ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# URLs to ingest
URLS = [
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/enhanced-cpf-housing-grant-families",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/cpf-housing-grants-for-resale-flats-families",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/step-up-cpf-housing-grant-families",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/proximity-housing-grant-families",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/seniors",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles/enhanced-cpf-housing-grant-singles",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles/cpf-housing-grant-for-resale-flats-singles",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles/proximity-housing-grant-singles",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/housing-loan-options/housing-loan-from-hdb",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/housing-loan-options/housing-loan-from-financial-institutions",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/application-for-an-hdb-flat-eligibility-hfe-letter",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/application-for-an-hdb-flat-eligibility-hfe-letter/income-guidelines"
]

RETRIEVER_K = 10 
RERANKER_TOP_N = 5

# Contextual Enrichment Prompt
CONTEXT_ENRICHMENT_PROMPT_TEMPLATE = """You are an intelligent assistant.

Document Title: {title}
Section: {section}

Chunk Content:
{content}

Please generate 1-2 sentences explaining what this specific chunk discusses in relation to the whole document.
"""

# HDBInfoAgent Prompt
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about public housing eligibility, loan types and application process. "
    "Use the following retrieved context to answer the question. "
    "If the context doesn't have the answer, say you don't find it in the documents."
    "\n\nContext:\n{context}"
)

# --- Retry Config ---
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5

# Query Expansion Prompt
QUERY_EXPANSION_PROMPT = (
    "You are an expert in information retrieval for Singapore public housing (HDB). "
    "Your goal is to generate 3 diverse and relevant search queries that are highly useful for finding information related to the user's question. "
    "These alternative queries should help retrieve a broader range of relevant documents. "
    "Provide ONLY the 3 queries, separated by newlines. Do not include any explanations or numbering."
    "\n\nUser Query: {question}"
)

# Router Prompt
ROUTER_PROMPT = (
    "You are an intelligent router for an HDB housing assistant. "
    "Classify the following user query into one of three categories:\n"
    "1. 'HDB_INFO': Queries related to HDB eligibility, housing loans, HFE application processes, grants, resale, or BTO.\n"
    "2. 'GREETING': Simple salutations, greetings, or thanks (e.g., 'hello', 'hi', 'thank you').\n"
    "3. 'UNRELATED': Queries completely off-topic (e.g., cooking, politics, general knowledge) that are not about HDB or housing in Singapore.\n\n"
    "Return ONLY the category name."
    "\n\nUser Query: {question}"
)

# Greeting Prompt
GREETING_PROMPT = (
    "You are a helpful and polite HDB assistant. "
    "Respond to the user's greeting warmly and truthfully. "
    "Briefly mention that you can help with HDB eligibility, loans, and HFE application processes. "
    "\n\nUser Input: {input}"
)

# Guardrail Generic Response
GUARDRAIL_MESSAGE = (
    "Sorry, but I am designed to assist specifically with HDB-related queries "
    "such as eligibility, housing loans, and the HFE application process. "
    "I cannot help with other topics."
)

GENERIC_ERROR_MESSAGE = "Apologizes, an unexpected error occurred. Please try again later."
