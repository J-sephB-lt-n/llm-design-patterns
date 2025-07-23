"""
Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database
"""

from pathlib import Path
from typing import Literal

import lancedb
import model2vec
import numpy as np
import openai
import pyarrow as pa
from lancedb.rerankers import RRFReranker
from loguru import logger

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


class VectorMemory(MemoryAlg):
    """
    Memory algorithm which stores only the few most recent chat messages, but also the full \
    chat history in a vector database

    Attributes:
        llm_client (openai.OpenAI): Model API client
        llm_name (str): Model name (model identifier in model API)
        llm_temperature (float): Model temperature (level of model output randomness)
        system_prompt (str): Preliminary instructions given to the language model
        n_chat_messages_in_working_memory (int): Number of most recent chat messages to keep in working 
                                            memory (model prompt).
                                            Messages older than this are embedded and written to 
                                            the vector database.
        n_vector_memories_to_fetch (int): In every chat completion, this number of chat messages will be 
                                            fetched from the full chat history (vector database) 
                                            and included in the model prompt.
        vector_search_method (str): Search algorithm used for fetching memories from the vector database
                                    One of ['semantic_dense', 'hybrid']
        vector_memory_type (str): Format used to generate vector memories.
                                    "chat_message" simply stores each chat message directly.
                                    "extracted_facts" uses an LLM to extract facts from each chat \
                                        message then embeds these facts and writes them to the db. 
        message_render_style (str): Controls how chat messages are rendered when including them in \
                                        model prompts.
                                        One of ['json_dumps', 'plain_text'].
                                        'plain_text' looks like "USER: ...<br>ASSISTANT: ..."
                                        'json_dumps' gives standard chat completion messages JSON
    """

    def __init__(
        self,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
        system_prompt: str = "You are a creative assistant helping a user to solve their problem.",
        n_chat_messages_in_working_memory: int = 10,
        n_vector_memories_to_fetch: int = 5,
        vector_search_method: Literal["semantic_dense", "hybrid"] = "hybrid",
        vector_memory_type: Literal["chat_message", "extracted_facts"] = "chat_message",
        message_render_style: Literal["plain_text", "json_dumps"] = "plain_text",
    ) -> None:
        self.chat_history: list[ChatMessageDetail] = []
        self.recent_chat_messages: list[ChatMessage] = []
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.system_prompt = system_prompt
        self.n_chat_messages_in_working_memory = n_chat_messages_in_working_memory
        self.n_vector_memories_to_fetch = n_vector_memories_to_fetch
        self.vector_search_method: Literal["semantic_dense", "hybrid"] = (
            vector_search_method
        )
        self.vector_memory_type: Literal["chat_message", "extracted_facts"] = (
            vector_memory_type
        )
        self.message_render_style: Literal["plain_text", "json_dumps"] = (
            message_render_style
        )

        self.embed_model = model2vec.StaticModel.from_pretrained(
            "minishlab/potion-retrieval-32M"
        )

        # set up temporary vector database on filesystem #
        self.vector_db = lancedb.connect(Path("temp_files/vector_memory_db"))
        self.vector_memories = self.vector_db.create_table(
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
        if self.vector_search_method == "hybrid":
            self.vector_memories.create_fts_index("text")
            self.vector_memories.wait_for_index("text_idx")
            self.reranker = RRFReranker()

    def add_vector_memory(self, text: str) -> None:
        """
        Write a memory into the vector database
        """
        self.vector_memories.add(
            [
                {
                    "text": text,
                    "vector": self.embed_model.encode(text),
                }
            ]
        )

    def fetch_relevant_memories(
        self,
        query: str,
        n_to_fetch: int,
        search_method: Literal["semantic_dense", "hybrid"],
    ) -> list[str]:
        """
        Fetch `n_to_fetch` closest memories to `query` from the vector database using `search_method`
        """
        embed_query: np.ndarray = self.embed_model.encode(query)
        match search_method:
            case "semantic_dense":
                search_results = (
                    self.vector_memories.search(embed_query).limit(n_to_fetch).to_list()
                )
            case "hybrid":
                search_results = (
                    self.vector_memories.vector(embed_query)
                    .text(query)
                    .rerank(self.reranker)
                    .limit(n_to_fetch)
                    .to_list()
                )
            case _:
                raise ValueError(f"Unknown search_method='{search_method}'")

        return [x["text"] for x in search_results]

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        self.recent_chat_messages.append(
            ChatMessage(
                role="user",
                message=user_msg,
            )
        )
        relevant_memories: list[str] = self.fetch_relevant_memories(
            query=self.chat_messages_to_text(
                messages=self.recent_chat_messages,
                output_style=self.message_render_style,
            ),
            n_to_fetch=self.n_vector_memories_to_fetch,
            search_method=self.vector_search_method,
        )
        prompt_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            self.augment_user_msg(user_msg, relevant_memories),
        ]
        logger.debug([msg.model_dump() for msg in prompt_messages])
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[msg.model_dump() for msg in self.recent_chat_messages],
        )

    def augment_user_msg(
        user_message: str,
        relevant_memories: list[str],
    ) -> ChatMessage:
        """ """
        ...

    def chat_messages_to_text(
        self,
        messages: list[ChatMessage],
        output_style: Literal["json_dumps", "plain_text"],
    ) -> str:
        """
        Represent sequence of chat-completion messages as a single string
        """
        match output_style:
            case "json_dumps":
                return json.dumps(
                    [msg.model_dump() for msg in messages],
                    indent=4,
                )
            case "plain_text":
                return "\n".join(
                    [f"{msg.role.upper()}: {msg.content}" for msg in messages]
                )
            case _:
                raise ValueError(f"Unknown output style '{output_style}'")
