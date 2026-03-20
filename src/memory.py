"""
Memory and State Management Module.

Demonstrates: Conversation history tracking, history-aware retrieval
using LangChain's create_history_aware_retriever, and manual
chat history management with HumanMessage/AIMessage.
"""

from typing import List, Tuple

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import ModelConfig
from src.prompts import build_contextualize_prompt
from src.chains import create_llm


class ConversationMemory:
    """
    Manages conversation state by tracking chat history as a list
    of HumanMessage and AIMessage objects.

    This approach gives full control over state management and integrates
    with LangChain's MessagesPlaceholder in prompt templates.
    """

    def __init__(self, max_turns: int = None):
        """
        Args:
            max_turns: Maximum number of conversation turns to retain.
                       Defaults to ModelConfig.MAX_HISTORY_LENGTH.
        """
        self.max_turns = max_turns or ModelConfig.MAX_HISTORY_LENGTH
        self.chat_history: List[BaseMessage] = []

    def add_user_message(self, message: str) -> None:
        """Record a user message."""
        self.chat_history.append(HumanMessage(content=message))
        self._trim()

    def add_ai_message(self, message: str) -> None:
        """Record an AI response."""
        self.chat_history.append(AIMessage(content=message))
        self._trim()

    def add_exchange(self, user_message: str, ai_message: str) -> None:
        """Record a full user-AI exchange."""
        self.add_user_message(user_message)
        self.add_ai_message(ai_message)

    def get_history(self) -> List[BaseMessage]:
        """Return the current chat history."""
        return self.chat_history

    def get_history_as_tuples(self) -> List[Tuple[str, str]]:
        """Return chat history as a list of (human, ai) tuples."""
        tuples = []
        for i in range(0, len(self.chat_history) - 1, 2):
            human = self.chat_history[i].content
            ai = self.chat_history[i + 1].content if i + 1 < len(self.chat_history) else ""
            tuples.append((human, ai))
        return tuples

    def clear(self) -> None:
        """Clear all conversation history."""
        self.chat_history = []
        print("[Memory] Conversation history cleared.")

    def _trim(self) -> None:
        """Trim history to max_turns (each turn = 2 messages)."""
        max_messages = self.max_turns * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

    def __len__(self) -> int:
        return len(self.chat_history) // 2


def create_history_aware_rag_retriever(retriever: VectorStoreRetriever):
    """
    Create a history-aware retriever that reformulates the user's question
    using chat history before performing retrieval.

    This manually implements the logic:
    1. Takes the chat history and current question.
    2. Uses an LLM to reformulate the question as standalone.
    3. Uses the reformulated question for retrieval.

    Args:
        retriever: The base vector store retriever.

    Returns:
        A history-aware retriever runnable.
    """
    llm = create_llm()
    contextualize_prompt = build_contextualize_prompt()

    # Create the reformulation chain: prompt | llm | parser
    reformulate_chain = contextualize_prompt | llm | StrOutputParser()

    # Create the history-aware retriever by combining reformulation with retrieval
    def history_aware_retriever_func(inputs):
        """
        Takes inputs with 'input' (question) and 'chat_history',
        reformulates the question, and retrieves documents.
        """
        # Reformulate the question using chat history
        reformulated_question = reformulate_chain.invoke({
            "input": inputs.get("input", ""),
            "chat_history": inputs.get("chat_history", []),
        })

        # Retrieve documents using the reformulated question
        docs = retriever.invoke(reformulated_question)
        return docs

    # Return as a Runnable
    return RunnableLambda(history_aware_retriever_func)
