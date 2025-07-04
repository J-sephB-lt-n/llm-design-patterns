"""
TODO
"""

from typing import Literal, Protocol

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatMessageDetail(BaseModel):
    """
    All messages within a single turn of the conversation

    Attributes:
        visible_messages (list[ChatMessage]): Messages visible to the chatting end-user
        all_messages (list[ChatMessage]): All messages involved in generating the chat \
            messages served to the chatting end-user (system messages, tool calls, memory \
            management by the LLM etc.)
    """

    visible_messages: list[ChatMessage]  # a subset of `all_messages`
    all_messages: list[ChatMessage]  # contains `visible_messages`


class MemoryAlg(Protocol):
    """
    Required behaviour of a chat-based memory algorithm
    """

    alg_name: str
    chat_history: list[ChatMessageDetail]

    def chat(self, str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        ...

    def view_memory_as_json(self) -> dict:
        """
        Render current state of the agent's memory as a dict
        """
        ...
