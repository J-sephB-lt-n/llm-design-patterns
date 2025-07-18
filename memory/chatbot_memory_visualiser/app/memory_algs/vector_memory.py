"""
Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database
"""

from typing import Literal

import openai

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


class VectorMemory(MemoryAlg):
    """
    Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database

    Attributes:
        llm_client (openai.OpenAI): Model API client
        llm_name (str): Model name (model identifier in model API)
        llm_temperature (float): Model temperature (level of model output randomness)
        n_chat_messages_in_working_memory (int): Number of most recent chat messages to keep in working 
                                            memory (model prompt).
                                            Messages older than this are embedded and written to 
                                            the vector database.
        n_vector_memories_to_fetch (int): In every chat completion, this number of chat messages will be 
                                            fetched from the full chat history (vector database) 
                                            and included in the model prompt.
        vector_embed_method (str): Approach used for embedding messages in long-term vector storage
                                    One of ['full_text_search', 'semantic_dense', 'hybrid']
    """

    def __init__(
        self,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
        n_chat_messages_in_working_memory: int = 10,
        n_vector_memories_to_fetch: int = 5,
        vector_embed_method: Literal[
            "full_text_search", "semantic_dense", "hybrid"
        ] = "hybrid",
    ) -> None:
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.n_chat_messages_in_working_memory = n_chat_messages_in_working_memory
        self.n_vector_memories_to_fetch = n_vector_memories_to_fetch
        self.vector_embed_method = vector_embed_method
