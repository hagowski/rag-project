"""
Document Processing Module.

Demonstrates: Document loading (multiple formats), text splitting
with RecursiveCharacterTextSplitter, and embedding generation
using OpenAI embeddings.
"""

import os
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import ModelConfig


class DocumentProcessor:
    """Handles document loading, splitting, and embedding creation."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ModelConfig.CHUNK_SIZE,
            chunk_overlap=ModelConfig.CHUNK_OVERLAP,
            separators=ModelConfig.SEPARATORS,
            length_function=len,
        )
        # Using local HuggingFace embeddings instead of OpenAI
        self.embeddings = HuggingFaceEmbeddings(
            model_name=ModelConfig.EMBEDDING_MODEL,
            cache_folder=os.path.join(os.path.dirname(__file__), "..", "models"),
            model_kwargs={"device": "cpu"},
        )

    def load_documents(self, directory: str = None) -> List[Document]:
        """
        Load documents from a directory. Supports .txt, .pdf, and .md files.

        Args:
            directory: Path to the documents directory.
                       Defaults to ModelConfig.DOCUMENTS_DIR.

        Returns:
            List of loaded Document objects.
        """
        directory = directory or ModelConfig.DOCUMENTS_DIR

        if not os.path.exists(directory):
            raise FileNotFoundError(f"Documents directory not found: {directory}")

        documents = []

        # Load .txt files
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        documents.extend(txt_loader.load())

        # Load .pdf files
        pdf_loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents.extend(pdf_loader.load())

        # Load .md files
        md_loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        )
        documents.extend(md_loader.load())

        if not documents:
            raise ValueError(f"No documents found in {directory}")

        print(f"[DocumentProcessor] Loaded {len(documents)} document(s).")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.

        Args:
            documents: List of Document objects to split.

        Returns:
            List of chunked Document objects.
        """
        chunks = self.text_splitter.split_documents(documents)
        print(
            f"[DocumentProcessor] Split into {len(chunks)} chunks "
            f"(chunk_size={ModelConfig.CHUNK_SIZE}, "
            f"overlap={ModelConfig.CHUNK_OVERLAP})."
        )
        return chunks

    def process(self, directory: str = None) -> List[Document]:
        """
        Full pipeline: load documents and split into chunks.

        Args:
            directory: Path to documents directory.

        Returns:
            List of chunked Document objects ready for embedding.
        """
        documents = self.load_documents(directory)
        chunks = self.split_documents(documents)
        return chunks

    def get_embeddings(self) -> Embeddings:
        """Return the configured embeddings model."""
        return self.embeddings
