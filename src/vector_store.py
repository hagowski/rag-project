"""
Vector Store Module.

Demonstrates: FAISS vector store creation from document chunks,
persistence (save/load), and similarity search retrieval.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config import ModelConfig


logger = logging.getLogger(__name__)


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_documents_checksum(documents_dir: str) -> str:
    """Compute a checksum of all documents in the directory."""
    doc_path = Path(documents_dir)
    if not doc_path.exists():
        return ""

    file_paths = []
    for ext in ["*.txt", "*.pdf", "*.md"]:
        file_paths.extend(doc_path.glob(ext))

    if not file_paths:
        return ""

    hashes = sorted([compute_file_hash(str(p)) for p in file_paths])
    combined = "".join(hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def get_per_file_hashes(documents_dir: str) -> dict:
    """Compute per-file content hashes for all documents in the directory.

    Returns:
        Dict mapping filename -> SHA256 hash of file content.
    """
    doc_path = Path(documents_dir)
    if not doc_path.exists():
        return {}

    file_hashes = {}
    for ext in ["*.txt", "*.pdf", "*.md"]:
        for p in doc_path.glob(ext):
            file_hashes[p.name] = compute_file_hash(str(p))

    return file_hashes


def detect_document_changes(documents_dir: str, stored_file_hashes: dict) -> dict:
    """Detect which documents have been added, modified, or removed.

    Args:
        documents_dir: Path to the documents directory.
        stored_file_hashes: Dict of filename -> hash from the last index build.

    Returns:
        Dict with keys 'added', 'modified', 'removed' (each a list of filenames),
        and 'has_changes' (bool).
    """
    current_hashes = get_per_file_hashes(documents_dir)
    stored_names = set(stored_file_hashes.keys())
    current_names = set(current_hashes.keys())

    added = sorted(current_names - stored_names)
    removed = sorted(stored_names - current_names)
    modified = sorted(
        name for name in (current_names & stored_names)
        if current_hashes[name] != stored_file_hashes[name]
    )

    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "has_changes": bool(added or modified or removed),
    }

class VectorStoreManager:
    """Manages FAISS vector store for document embeddings."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        # Accept any object that provides an `embed_documents` method (e.g., OpenAIEmbeddings,
        # a dummy stub, or another compatible provider).
        self.embeddings = embeddings
        self.vector_store: FAISS = None

    def create_from_documents(self, chunks: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from document chunks.

        Each chunk is embedded using the configured OpenAI embedding model
        and stored in a FAISS index for efficient similarity search.

        Args:
            chunks: List of chunked Document objects.

        Returns:
            FAISS vector store instance.
        """
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        logger.info(f"Created FAISS index with {len(chunks)} vectors.")
        return self.vector_store

    def save(self, path: str = None) -> None:
        """Persist the vector store to disk."""
        path = path or ModelConfig.VECTOR_STORE_DIR
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        os.makedirs(path, exist_ok=True)
        self.vector_store.save_local(path)
        logger.info(f"Saved vector store to {path}.")

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Get a retriever interface for the vector store.

        Returns:
            A VectorStoreRetriever configured with top-k and search type
            from ModelConfig.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")

        return self.vector_store.as_retriever(
            search_type=ModelConfig.SEARCH_TYPE,
            search_kwargs={"k": ModelConfig.TOP_K_RESULTS},
        )

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform a direct similarity search.

        Args:
            query: The search query string.
            k: Number of results to return (defaults to ModelConfig.TOP_K_RESULTS).

        Returns:
            List of the most similar Document objects.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available.")

        k = k or ModelConfig.TOP_K_RESULTS
        results = self.vector_store.similarity_search(query, k=k)
        print(f"[VectorStore] Found {len(results)} results for query.")
        return results

    def save_with_checksum(self, path: str = None) -> None:
        """Save vector store and store document checksum + per-file hashes for change detection."""
        path = path or ModelConfig.VECTOR_STORE_DIR
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")

        os.makedirs(path, exist_ok=True)

        # Save the actual vector store
        self.vector_store.save_local(path)

        # Compute and save checksum + per-file hashes + build timestamp
        checksum = get_documents_checksum(ModelConfig.DOCUMENTS_DIR)
        file_hashes = get_per_file_hashes(ModelConfig.DOCUMENTS_DIR)
        meta_file = os.path.join(path, "metadata.json")
        with open(meta_file, "w") as f:
            json.dump({
                "documents_checksum": checksum,
                "file_hashes": file_hashes,
                "build_timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        logger.info(f"Saved vector store and metadata to {path}.")

    def load(self, path: str = None) -> FAISS:
        """Load a persisted vector store from disk.

        Raises FileNotFoundError with change details if documents have changed.
        """
        path = path or ModelConfig.VECTOR_STORE_DIR
        if not os.path.exists(path):
            raise FileNotFoundError(f"No vector store found at {path}.")

        index_file = os.path.join(path, "index.faiss")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"No vector store found at {path}.")

        # Check if documents have changed since last build
        metadata_file = os.path.join(path, "metadata.json")
        rebuild_required = False
        change_details = None

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                stored_meta = json.load(f)

            current_checksum = get_documents_checksum(ModelConfig.DOCUMENTS_DIR)
            stored_checksum = stored_meta.get("documents_checksum", "")

            if current_checksum != stored_checksum:
                # Detect per-file changes
                stored_file_hashes = stored_meta.get("file_hashes", {})
                change_details = detect_document_changes(
                    ModelConfig.DOCUMENTS_DIR, stored_file_hashes
                )
                rebuild_required = True
        else:
            # No metadata file means old version - rebuild to add metadata
            rebuild_required = True

        try:
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.error(f"Corrupt or unreadable vector store at {path}: {e}")
            raise FileNotFoundError(f"Could not load vector store at {path}: {e}")

        if rebuild_required:
            msg = "Documents changed"
            if change_details:
                parts = []
                if change_details["added"]:
                    parts.append(f"added: {', '.join(change_details['added'])}")
                if change_details["modified"]:
                    parts.append(f"modified: {', '.join(change_details['modified'])}")
                if change_details["removed"]:
                    parts.append(f"removed: {', '.join(change_details['removed'])}")
                if parts:
                    msg += f" ({'; '.join(parts)})"
            raise FileNotFoundError(msg)

        logger.info(f"Loaded vector store from {path}.")
        return self.vector_store

    def get_change_status(self, path: str = None) -> dict:
        """Check document change status without loading the store.

        Returns:
            Dict with change detection results and stored metadata.
        """
        return get_change_status(path)


def get_change_status(path: str = None) -> dict:
    """Standalone change detection — no embeddings required.

    Returns:
        Dict with change detection results and stored metadata.
    """
    path = path or ModelConfig.VECTOR_STORE_DIR
    metadata_file = os.path.join(path, "metadata.json")

    if not os.path.exists(metadata_file):
        return {"status": "no_metadata", "has_changes": True}

    with open(metadata_file, "r") as f:
        stored_meta = json.load(f)

    stored_file_hashes = stored_meta.get("file_hashes", {})
    changes = detect_document_changes(ModelConfig.DOCUMENTS_DIR, stored_file_hashes)
    changes["status"] = "checked"
    changes["indexed_files"] = sorted(stored_file_hashes.keys())
    return changes
