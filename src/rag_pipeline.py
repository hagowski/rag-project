"""
RAG Pipeline Module.

Orchestrates all components: document processing, vector store,
prompts, chains, and memory into a unified RAG system.
"""
import os
from config import ModelConfig
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager, get_documents_checksum, detect_document_changes
from src.memory import ConversationMemory, create_history_aware_rag_retriever
from src.chains import create_conversational_rag_chain


class RAGPipeline:
    """
    Main RAG pipeline that ties together:
    - Document processing and embedding
    - Vector store creation and retrieval
    - Prompt composition
    - LCEL chaining
    - Conversation memory and state management
    """

    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager(self.doc_processor.get_embeddings())
        self.memory = ConversationMemory()
        self.chain = None
        self._initialized = False

    def initialize(self, documents_dir: str = None) -> None:
        """
        Initialize the pipeline: load documents, create embeddings,
        build the vector store, and assemble the chain.

        Checks for per-file document changes and logs which files triggered
        a rebuild.

        Args:
            documents_dir: Optional path to documents directory.
        """
        print("\n=== Initializing RAG Pipeline ===\n")

        # Step 1: Try to load the vector store from disk
        try:
            self.vector_manager.load()
            print("[VectorStore] Loaded existing index. No document changes detected.")
        except FileNotFoundError as e:
            error_msg = str(e)
            documents_dir = documents_dir or ModelConfig.DOCUMENTS_DIR

            if "Documents changed" in error_msg:
                print(f"[VectorStore] {error_msg}")
                print("[VectorStore] Rebuilding index...")
            else:
                print("[VectorStore] No existing store found. Building a new one...")

            chunks = self.doc_processor.process(documents_dir)
            self.vector_manager.create_from_documents(chunks)
            self.vector_manager.save_with_checksum()
            print(f"[VectorStore] Index built with {len(chunks)} chunks.")

        # Step 2: Get retriever from the now-loaded vector store
        retriever = self.vector_manager.get_retriever()

        # Step 3: Create history-aware retriever (reformulates questions)
        history_aware_retriever = create_history_aware_rag_retriever(retriever)

        # Step 4: Build the full conversational chain from our chains module
        self.chain = create_conversational_rag_chain(history_aware_retriever)

        self._initialized = True
        print("\n=== RAG Pipeline Ready ===\n")

    def ask(self, question: str) -> str:
        """
        Ask a question using the RAG pipeline with conversation memory.

        Args:
            question: The user's question.

        Returns:
            The AI-generated answer.
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Invoke the chain with the question and chat history
        response = self.chain.invoke({
            "question": question,
            "chat_history": self.memory.get_history(),
        })

        # Update memory with the new exchange
        self.memory.add_exchange(question, response)

        return response

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()

    def get_history_length(self) -> int:
        """Return the number of conversation turns."""
        return len(self.memory)

    def search_documents(self, query: str, k: int = None):
        """
        Directly search the vector store (useful for debugging).

        Args:
            query: Search query.
            k: Number of results.

        Returns:
            List of matching Document objects.
        """
        return self.vector_manager.similarity_search(query, k)
