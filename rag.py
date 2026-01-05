import os

import logging
from typing import Tuple, List, Generator, Any, Sequence

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

import config
from utils import setup_logger, format_docs

logger = setup_logger("rag_pipeline")


# --- Agents Definition ---

class BaseAgent:
    def stream_response(self, input_text: str) -> Generator[Tuple[str, List[Document]], None, None]:
        raise NotImplementedError("Subclasses must implement stream_response")

class GreetingAgent(BaseAgent):
    def __init__(self, llm):
        self.llm = llm
        
    def stream_response(self, input_text: str) -> Generator[Tuple[str, List[Document]], None, None]:
        logger.info("[GreetingAgent] Activated.")
        yield "", []
        
        prompt = config.GREETING_PROMPT.format(input=input_text)
        try:
            for chunk in self.llm.stream(prompt):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                yield content, []
        except Exception as e:
            logger.error(f"[GreetingAgent] Error: {e}")
            yield config.GENERIC_ERROR_MESSAGE, []


class GuardrailAgent(BaseAgent):
    # just return the standard response
    def stream_response(self, input_text: str) -> Generator[Tuple[str, List[Document]], None, None]:
        logger.info("[GuardrailAgent] Activated.")
        yield config.GUARDRAIL_MESSAGE, []


class HDBInfoAgent(BaseAgent):
    def __init__(self, llm, vectorstore, device):
        self.llm = llm
        self.vectorstore = vectorstore
        self.device = device
        self.compressor = None
        self.retriever = self._build_retriever()
        self.qa_chain = self._build_chain()
        
    def _build_retriever(self):
        logger.info("[HDBInfoAgent] Building Retriever...")
        dense_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVER_K,
                "fetch_k": config.RETRIEVER_K * 2,
                "lambda_mult": 0.5,
            }
        )
        
        # Sparse (BM25)
        # Reconstruct documents for BM25
        data = self.vectorstore.get()
        docs_list = []
        if data['documents']:
            for i, text in enumerate(data['documents']):
                meta = data['metadatas'][i] if data['metadatas'] else {}
                docs_list.append(Document(page_content=text, metadata=meta))
        
        if not docs_list:
            sparse_retriever = dense_retriever
        else:
            sparse_retriever = BM25Retriever.from_documents(docs_list)
            sparse_retriever.k = config.RETRIEVER_K
            
        # Ensemble
        base_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.6, 0.4] 
        )
        
        # Reranker
        model = HuggingFaceCrossEncoder(
            model_name=config.CROSS_ENCODER_MODEL_NAME,
            model_kwargs={'device': self.device}
        )
        self.compressor = CrossEncoderReranker(model=model, top_n=config.RERANKER_TOP_N)
        
        final_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=base_retriever
        )
        return final_retriever

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.RAG_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        return prompt | self.llm

    def expand_query(self, question: str) -> List[str]:
        logger.info("[HDBInfoAgent] Expanding query...")
        prompt = config.QUERY_EXPANSION_PROMPT.format(question=question)
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            # Cleanup
            cleaned_queries = []
            for query in queries:
                 if query[0].isdigit() and '. ' in query[:4]: query = query.split('. ', 1)[1]
                 elif query.startswith('- '): query = query[2:]
                 cleaned_queries.append(query)
            logger.info(f"[HDBInfoAgent] Expanded queries: {cleaned_queries}")
            return cleaned_queries
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []

    def _retrieve_with_expansion(self, question: str) -> List[Document]:
        expanded_queries = self.expand_query(question)[:3] # Ensure max 3
        search_queries = [question] + expanded_queries
        logger.info(f"[HDBInfoAgent] Retrieving documents for {len(search_queries)} queries...")
        
        all_docs = []
        ensemble = self.retriever.base_retriever
        
        for i, q in enumerate(search_queries):
             try:
                 logger.debug(f"[HDBInfoAgent] Retrieving for query {i+1}: {q}")
                 docs = ensemble.invoke(q)
                 all_docs.extend(docs)
             except Exception as e:
                 logger.warning(f"Retrieval failed for {q}: {e}")

        logger.info(f"[HDBInfoAgent] Total docs retrieved (pre-dedup): {len(all_docs)}")

        # Dedup in case the same doc 
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        deduped = list(unique_docs.values())
        logger.info(f"[HDBInfoAgent] Unique docs after dedup: {len(deduped)}")
        
        # Rerank against ORIGINAL question
        if deduped:
            logger.info("[HDBInfoAgent] Starting reranking...")
            try:
                reranked = self.compressor.compress_documents(deduped, question)
                logger.info(f"[HDBInfoAgent] Reranking complete. Top {len(reranked)} docs selected.")
            except Exception as e:
                logger.error(f"[HDBInfoAgent] Reranking failed: {e}")
                reranked = deduped[:config.RETRIEVER_K] # Fallback
        else:
            reranked = []
            
        return reranked

    def stream_response(self, input_text: str) -> Generator[Tuple[str, List[Document]], None, None]:
        logger.info(f"[HDBInfoAgent] Processing: {input_text}")
        
        docs = self._retrieve_with_expansion(input_text)
        yield "", docs
        
        logger.info("[HDBInfoAgent] Starting generation...")    
        try:
            # Format docs to string
            context_str = format_docs(docs)
            for chunk in self.qa_chain.stream({"input": input_text, "context": context_str}):
                if isinstance(chunk, str): yield chunk, docs
                elif hasattr(chunk, 'content'): yield chunk.content, docs
                else: yield str(chunk), docs
            logger.info("[HDBInfoAgent] Generation completed successfully.")
        except Exception as e:
            logger.error(f"[HDBInfoAgent] Generation Error: {e}")
            yield config.GENERIC_ERROR_MESSAGE, docs


class MainAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def classify_intent(self, question: str) -> str:
        try:
            prompt = config.ROUTER_PROMPT.format(question=question)
            response = self.llm.invoke(prompt)
            intent = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            if "HDB_INFO" in intent: return "HDB_INFO"
            if "GREETING" in intent: return "GREETING"
            if "UNRELATED" in intent: return "UNRELATED"
            
            logger.warning(f"Unclear intent '{intent}', defaulting to HDB_INFO")
            return "HDB_INFO"
        except Exception as e:
            logger.error(f"Intent Error: {e}")
            return "HDB_INFO"


class RAGPipeline:
    def __init__(self):
        logger.info("Initializing Agentic RAG Pipeline...")
        self.device = "cuda"
        
        # Shared Resources
        self.vectorstore = self._load_vectorstore()
        self.llm = self._load_llm()
        
        # Initialize Agents
        self.main_agent = MainAgent(self.llm)
        self.greeting_agent = GreetingAgent(self.llm)
        self.guardrail_agent = GuardrailAgent()
        self.hdb_agent = HDBInfoAgent(self.llm, self.vectorstore, self.device)
        
        logger.info("Agentic RAG Pipeline initialized.")

    def _load_vectorstore(self) -> Chroma:
        logger.info(f"Loading Vector Store from: {config.CHROMA_PATH}")
        if not os.path.exists(config.CHROMA_PATH):
             raise ValueError("Vector store not found.")
        embedding_function = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': self.device, 'trust_remote_code': True}
        )
        return Chroma(persist_directory=config.CHROMA_PATH, embedding_function=embedding_function)

    def _load_llm(self) -> ChatOllama:
        return ChatOllama(model=config.MODEL_ID, temperature=0.1, timeout=120) #can adjust the timeout if needed

    def stream_query(self, question: str) -> Generator[Tuple[str, List[Document]], None, None]:
        """
        Generator that yields (token, docs).
        Delegates to the appropriate Agent based on intent.
        """
        logger.info(f"Streaming query: {question}")
        
        # Check intent and routing
        intent = self.main_agent.classify_intent(question)
        logger.info(f"Main Agent detected Intent: {intent}")
        
        # Delegation
        if intent == "GREETING":
            yield from self.greeting_agent.stream_response(question)
        elif intent == "UNRELATED":
            yield from self.guardrail_agent.stream_response(question)
        else: # HDB_INFO or fallback
            yield from self.hdb_agent.stream_response(question)

    def query(self, question: str) -> Tuple[str, List[Document]]:
        # Sync wrapper
        text = []
        docs = []
        for token, d in self.stream_query(question):
            if d: docs = d
            if token: text.append(token)
        return "".join(text), docs

if __name__ == "__main__":
    rag = RAGPipeline()
    print("Testing Agentic Flow...")
    for q in ["Hello", "Can I buy a flat if im 17 years old?", "Recipe for cake"]:
        print(f"\n--- Query: {q} ---")
        ans = ""
        for t, _ in rag.stream_query(q):
            print(t, end="", flush=True)
            ans += t
        print("\n")



