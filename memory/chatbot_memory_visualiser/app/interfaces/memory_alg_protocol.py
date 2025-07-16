"""
Required interface which all memory algorithms must follow to integrate with this app
"""

from typing import Literal, Protocol

import openai
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatMessageDetail(BaseModel):
    """
    All messages within a single turn of the conversation (a single chat completion)

    Attributes:
        visible_messages (list[ChatMessage]): Messages visible to the chatting end-user
        all_messages (list[ChatMessage]): All messages involved in generating the chat \
            messages served to the chatting end-user (system messages, tool calls, memory \
            management by the LLM etc.)
        token_usage (dict): Token usage breakdown consumed by this single chat completion
    """

    visible_messages: list[ChatMessage]  # a subset of `all_messages`
    all_messages: list[ChatMessage]  # contains `visible_messages`
    token_usage: dict


class MemoryAlg(Protocol):
    """
    Required behaviour of a chat-based memory algorithm
    """

    llm_client: openai.OpenAI
    chat_history: list[ChatMessageDetail]
    alg_description: str

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        ...

    def view_memory_as_json(self) -> dict:
        """
        Render latest state of the agent's memory as a dict
        """
        ...
