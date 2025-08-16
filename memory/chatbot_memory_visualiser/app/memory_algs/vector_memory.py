"""
Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database.
"""

import json
import re
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
from app.lifecycle.teardown import app_cleanup


class VectorMemory(MemoryAlg):
    """
    Memory algorithm which stores only the few most recent chat messages, but also the full \
    chat history in a vector database

    Note:
        - The way that memories are currently written to vector storage cuts off the conversation \
          in an arbitary place (i.e. a logical multi-turn segment of the conversation might be cut in half).
        - In a production implementation of this algorithm, a rolling window approach (with dedup at \
          retrieval time) would be better, so that every conversation window is captured.
        - If facts are being extracted (rather than just conversation snippets saved), then the same \
          thing applies (except there will need to be a LLM fact deduping step).

    Algorithm summary:
        1. The last `n_chat_messages_in_working_memory` most recent chat messages are kept in context
            (i.e. in the prompt)
        2. To reply to a new user message, the `n_vector_memories_to_fetch` most similar memories to the \
            `recall_query_n_chat_messages` most recent chat messages are retrieved from the vector database
        3. The assistant generates a response based on the user message, the recent chat history and the \
            retrieved memories
        4. When the recent chat history gets `n_chat_messages_in_working_memory` message long, the \
            `archive_n_messages_at_a_time` oldest chat messages are written to archival storage (the vector \
            database).
            This is done either by:
                a) Embedding the content of these messages directly (just the chat messages)
                    OR
                b) The LLM extracts facts from these messages and the facts are written to the vector database
            Which approach is used is controlled by `vector_memory_type`

    Attributes:
        llm_client (openai.OpenAI): Model API client
        llm_name (str): Model name (model identifier in model API)
        llm_temperature (float): Model temperature (level of model output randomness)
        system_prompt (str): Preliminary instructions given to the language model
        n_chat_messages_in_working_memory (int): Number of most recent chat messages to keep in working 
            memory (model prompt).
            Messages older than this are embedded and written to the vector database.
        n_vector_memories_to_fetch (int): In every chat completion, this number of memories are
            fetched from the vector database and included in the model prompt.
        recall_query_n_chat_messages (int): The number of (most recent) messages in the recent chat history 
            to use as query when fetching from the vector database.
        archive_n_messages_at_a_time (int): Number of messages to group together when removing messages 
            from the recent chat history and writing them to archival storage (the vector database)
        vector_search_method (str): Search algorithm used for fetching memories from the vector database
            One of ['semantic_dense', 'hybrid']
        vector_memory_type (str): Format used to generate vector memories.
            "chat_message" simply stores each chat message directly.
            "extracted_facts" uses an LLM to extract facts from each chat message then \
            embeds these facts and writes them to the vector database. 
        message_render_style (str): Controls how chat messages are rendered when including them in \
            model prompts.
            One of ['json_dumps', 'plain_text', 'xml'].
            'plain_text' looks like "USER: ...<br>ASSISTANT: ...<br> ..."
            'json_dumps' gives standard chat completion messages JSON
            'xml' looks like "<user>...</user> <assistant>...</assistant> ..."
    """

    def __init__(
        self,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
        system_prompt: str = "You are a creative assistant helping a user to solve their problem.",
        n_chat_messages_in_working_memory: int = 10,
        n_vector_memories_to_fetch: int = 5,
        recall_query_n_chat_messages: int = 1,
        archive_n_messages_at_a_time: int = 4,
        vector_search_method: Literal["semantic_dense", "hybrid"] = "hybrid",
        vector_memory_type: Literal[
            "chat_message", "extracted_facts"
        ] = "extracted_facts",
        message_render_style: Literal["plain_text", "json_dumps", "xml"] = "plain_text",
    ) -> None:
        # delete any temporary files created by previous alg #
        app_cleanup()

        if recall_query_n_chat_messages > n_chat_messages_in_working_memory:
            raise ValueError("`recall_query_n_chat_messages` must be smaller than `n_chat_messages_in_working_memory`")

        if archive_n_messages_at_a_time > n_chat_messages_in_working_memory:
            raise ValueError("`archive_n_messages_at_a_time` must be smaller than `n_chat_messages_in_working_memory`")

        self.chat_history: list[ChatMessageDetail] = []
        self.recent_chat_messages: list[ChatMessage] = []
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.system_prompt = system_prompt
        self.n_chat_messages_in_working_memory = n_chat_messages_in_working_memory
        self.n_vector_memories_to_fetch = n_vector_memories_to_fetch
        self.recall_query_n_chat_messages = recall_query_n_chat_messages
        self.archive_n_messages_at_a_time = archive_n_messages_at_a_time
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
                    self.vector_memories.search(query_type="hybrid")
                    .vector(embed_query)
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
                content=user_msg,
            )
        )
        memory_query: str = self.chat_messages_to_text(
                messages=self.recent_chat_messages[-self.recall_query_n_chat_messages:],
                message_render_style=self.message_render_style,
            )
        logger.debug(
            f"--Memory retrieval query--\n{memory_query}" 
        )
        relevant_memories: list[str] = self.fetch_relevant_memories(
            query=memory_query,
            n_to_fetch=self.n_vector_memories_to_fetch,
            search_method=self.vector_search_method,
        )
        prompt_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            self.augment_user_msg(user_msg, relevant_memories),
        ]
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[msg.model_dump() for msg in prompt_messages],
        )
        assistant_response: ChatMessage = ChatMessage(
            role=llm_api_response.choices[0].message.role,
            content=llm_api_response.choices[0].message.content,
        )
        logger.debug(
            "\n".join(
                f"--{msg.role.upper()}--\n{msg.content}" 
                for msg in prompt_messages + [assistant_response]
            )
        )
        self.recent_chat_messages.append(assistant_response)
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[
                    ChatMessage(role="user", content=user_msg),
                    assistant_response,
                ],
                all_messages=prompt_messages + [assistant_response],
                token_usage=llm_api_response.usage.model_dump(),
            )
        )
        if len(self.recent_chat_messages) > self.n_chat_messages_in_working_memory:
            messages_to_archive: list[ChatMessage] = self.recent_chat_messages[:self.archive_n_messages_at_a_time]
            self.recent_chat_messages = self.recent_chat_messages[self.archive_n_messages_at_a_time:]
            match self.vector_memory_type:
                case "chat_message":
                    self.add_vector_memory(
                        self.chat_messages_to_text(
                            messages=messages_to_archive,
                            message_render_style=self.message_render_style,
                        )
                    )
                case "extracted_facts":
                    facts: list[str] = self.extract_facts_from_chat_snippet(
                        messages_to_archive
                    )
                    for fact in facts:
                        self.add_vector_memory(fact)

    def augment_user_msg(
        self,
        user_message: str,
        relevant_memories: list[str],
    ) -> ChatMessage:
        """
        Add context (recent chat history and retrieved long-term memories) to user message
        """
        return ChatMessage(
            role="user",
            content=f"""
<possibly-relevant-memories>
{"\n".join(str(num) + ". " + memory for num, memory in enumerate(relevant_memories, start=1))}
</possibly-relevant-memories>

<recent-chat-history>
{self.chat_messages_to_text(self.recent_chat_messages[:-1], self.message_render_style)}
</recent-chat-history>

<user-query>
{user_message}
</user-query>
""".strip(),
        )

    def extract_facts_from_chat_snippet(
        self,
        messages=list[ChatMessage],
    ) -> list[str]:
        """
        Extract facts relevant to the conversation from a subset of a chat
        """
        prompt: str = f"""
<conversation-snippet>
{self.chat_messages_to_text(messages=messages, message_render_style=self.message_render_style)}
</conversation-snippet>

From the given conversation snippet, extract all facts (distinct pieces of knowledge). \
These facts will be retrieved by the assistant (you) in order to generate informed \
replies in future conversations with this user.

Return the facts as a list of strings contained within a JSON markdown code block. If there \
are no facts, return an empty list.

<example-output-format>
```json
["fact 1 here", "fact 2 here", ...]
```
</example-output-format>
                """
        logger.debug(prompt)
        prompt_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=prompt,
            ),
        ]
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[msg.model_dump() for msg in prompt_messages],
        )
        logger.debug(llm_api_response.choices[0].message.content)
        extracted_facts: list[str] = json.loads(
            re.search(
                r"```json\s*\n(?P<facts>.*?)\n```",
                llm_api_response.choices[0].message.content,
                re.DOTALL,
            ).group("facts"),
        )
        logger.debug(json.dumps(extracted_facts, indent=4))
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[],
                all_messages=prompt_messages
                + [
                    ChatMessage(
                        role="assistant",
                        content=llm_api_response.choices[0].message.content,
                    )
                ],
                token_usage=llm_api_response.usage.model_dump(),
            )
        )
        return extracted_facts

    def chat_messages_to_text(
        self,
        messages: list[ChatMessage],
        message_render_style: Literal["json_dumps", "plain_text", "xml"],
    ) -> str:
        """
        Represent sequence of chat-completion messages as a single string
        """
        match message_render_style:
            case "json_dumps":
                return json.dumps(
                    [msg.model_dump() for msg in messages],
                    indent=4,
                )
            case "plain_text":
                return "\n".join(
                    [f"{msg.role.upper()}: {msg.content}" for msg in messages]
                )
            case "xml":
                return "\n".join(
                    [f"<{msg.role}>\n{msg.content}\n</{msg.role}>" for msg in messages]
                )

    def view_memory_as_json(self) -> dict:
        """
        Render latest state of the agent's memory as a dict
        """
        return {
            "recent_chat_history": [
                msg.model_dump() for msg in self.recent_chat_messages
            ],
            "long_term_chat_history": [
                x["text"] for x in self.vector_memories.search().to_list()
            ],
        }
