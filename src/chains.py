"""
Chaining Mechanisms Module.

Demonstrates: LangChain Expression Language (LCEL), pipe operator (|),
RunnablePassthrough, RunnableParallel, and retrieval chain construction.
"""

import os

from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel

from config import ModelConfig
from src.prompts import build_rag_prompt


def create_llm() -> ChatOpenAI:
    """
    Create a configured ChatOpenAI LLM instance.

    Demonstrates model configuration and output control:
    - model selection
    - temperature control
    - max token limit
    - streaming toggle
    """
    return ChatOpenAI(
        model=ModelConfig.LLM_MODEL,
        temperature=ModelConfig.TEMPERATURE,
        max_tokens=ModelConfig.MAX_TOKENS,
        streaming=ModelConfig.STREAMING,
        openai_api_key=ModelConfig.get_api_key(),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
    )


def format_docs(docs) -> str:
    """Format a list of documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_conversational_rag_chain(history_aware_retriever: Runnable) -> Runnable:
    """
    Build the full conversational RAG chain.

    This chain orchestrates history-aware retrieval with question answering.

    Pipeline:
      1. The user's question and chat history are passed to the
         `history_aware_retriever`.
      2. The retriever reformulates the question and fetches relevant documents.
      3. The documents, original question, and history are passed to a
         final prompt and LLM to generate the answer.

    Args:
        history_aware_retriever: A runnable that takes a question and
                                 history and returns documents.

    Returns:
        The complete, runnable conversational RAG chain.
    """
    llm = create_llm()
    prompt = build_rag_prompt()

    # This is the final chain that answers the question based on context
    question_answer_chain = prompt | llm | StrOutputParser()

    # This is the full conversational chain
    return (
        RunnableParallel(
            # The history_aware_retriever is invoked here with the correct input
            context=lambda x: format_docs(
                history_aware_retriever.invoke(
                    {"input": x["question"], "chat_history": x.get("chat_history", [])}
                )
            ),
            # The original question and history are passed through
            question=lambda x: x["question"],
            chat_history=lambda x: x.get("chat_history", []),
            current_date=lambda x: datetime.now().strftime("%Y-%m-%d"),
        )
        | question_answer_chain
    )
