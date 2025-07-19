"""
Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database
"""

from pathlib import Path
from typing import Literal

import lancedb
import model2vec
import openai
import pyarrow as pa

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


class VectorMemory(MemoryAlg):
    """
    Memory algorithm which stores only the few most recent chat messages, but also the full \
    chat history in a vector database

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
        vector_memory_type (str): Format used to generate vector memories.
                                    "chat_message" simply stores each chat message directly.
                                    "extracted_facts" uses an LLM to extract facts from each chat \
                                        message then embeds these facts and writes them to the db. 
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
        vector_memory_type: Literal["chat_message", "extracted_facts"] = "chat_message",
    ) -> None:
        self.chat_history: list[ChatMessageDetail] = []
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.n_chat_messages_in_working_memory = n_chat_messages_in_working_memory
        self.n_vector_memories_to_fetch = n_vector_memories_to_fetch
        self.vector_embed_method = vector_embed_method
        self.vector_memory_type = vector_memory_type

        self.embed_model = model2vec.StaticModel.from_pretrained(
            "minishlab/potion-retrieval-32M"
        )
        self.vector_db = lancedb.connect(Path("temp_files/vector_memory_db"))
        self.vector_db.create_table(
            name="memories",
            schema=pa.schema(
                [
                    pa.field("text", pa.string()),
                    pa.field(
                        "vector",
                        pa.list_(
                            pa.float32(),
                            512,
                        ),
                    ),
                ],
            ),
        )

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        ...
