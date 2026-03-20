"""
Prompt Templates and Composition Module.

Demonstrates: PromptTemplate, ChatPromptTemplate, SystemMessage,
HumanMessage, partial variables, and prompt composition by combining
multiple prompt templates into a single pipeline prompt.
"""

from datetime import datetime

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


# ---------------------------------------------------------------------------
# 1. Basic Prompt Templates
# ---------------------------------------------------------------------------

# Simple question-answer prompt template
QA_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# ---------------------------------------------------------------------------
# 2. Chat Prompt Templates with System / Human messages
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = (
    "You are a helpful assistant that answers questions based on the "
    "provided context. Today's date is {current_date}.\n\n"
    "Guidelines:\n"
    "- Only answer based on the provided context.\n"
    "- If the context does not contain the answer, say so honestly.\n"
    "- Be concise and accurate.\n"
    "- Cite relevant parts of the context when possible."
)

HUMAN_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# ---------------------------------------------------------------------------
# 3. Prompt Composition - combining templates into a full chat prompt
# ---------------------------------------------------------------------------

def build_rag_prompt() -> ChatPromptTemplate:
    """
    Build the main RAG prompt using prompt composition.

    Combines:
      - A system message template (with a partial variable for the date)
      - A messages placeholder for chat history (memory)
      - A human message template for context + question

    Returns:
        A composed ChatPromptTemplate.
    """
    # Create individual message templates
    # System message with a placeholder for the current date; it will be filled at runtime via partial variables.
    system_msg = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
    # No formatting here – keep {current_date} as a variable to be supplied when rendering.

    human_msg = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

    # Compose: system + history placeholder + human
    prompt = ChatPromptTemplate.from_messages([
        system_msg,
        MessagesPlaceholder(variable_name="chat_history"),
        human_msg,
    ])

    return prompt


# ---------------------------------------------------------------------------
# 4. Contextualisation Prompt (for history-aware retrieval)
# ---------------------------------------------------------------------------

CONTEXTUALIZE_SYSTEM_TEMPLATE = (
    "Given the chat history and the latest user question, "
    "reformulate the question to be a standalone question that "
    "can be understood without the chat history. "
    "Do NOT answer the question — only reformulate it if needed, "
    "otherwise return it as-is."
)


def build_contextualize_prompt() -> ChatPromptTemplate:
    """
    Build a prompt that reformulates a user question using chat history,
    so the retriever can find relevant documents without needing the
    full conversation context.

    Returns:
        ChatPromptTemplate for question contextualisation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
