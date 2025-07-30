"""
Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database
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


class KnowledgeGraphMemory(MemoryAlg):
    """
    Memory algorithm which stores memories as a knowledge graph

    More specifically, the algorithm works as follows:
        - At each new user message in the chat, a LLM is used to extract semantic (RDF) triples \
from the last `triples_source_n_msgs` chat messages in the conversation
        - The subject, predicate and objects text content are cleaned, to prevent duplication \
in the graph (strip, lowercase etc.)
        - These new triples are deduped against the existing knowledge triples already in the \
graph using a LLM to make the decisions
        - Triples which are new (i.e. not duplicates) are added to the graph - the subject and \
object as nodes and the predicate as an edge.
        - The content of the new nodes (subject and object) is embedded and added into a vector \
database (for later lookup)
        - Each new user chat message is augmented with knowledge context as follows:
            - The nearest `n_context_nodes` nodes whose content is most similar to the combined \
content of the last `context_n_messages` chat messages are retrieved.
            - For each of these retrieved nodes, the graph is traversed by `n_context_hops` \
steps, and all of the knowledge triples explored this way are included as context in the prompt.

    Attributes:
        llm_client (openai.OpenAI): Model API client
        llm_name (str): Model name (model identifier in model API)
        llm_temperature (float): Model temperature (level of model output randomness)
        system_prompt (str): Preliminary instructions given to the language model
        triples_source_n_messages (int): Number of most recent chat messages to use to extract \
knowledge triples
        n_context_nodes (int): Number of initial nodes fetched from the knowledge graph when \
adding knowledge context to the prompt
        context_n_messages (int): Number of most recent chat messages to use in as query when \
fetching most relevant nodes from the graph 
        n_context_hops (int): Number of steps to take when traversing the graph to find \
possibly related knowledge, outward from the first `n_context_nodes` nodes found.
        vector_search_method (str): Approach used for fetching nodes from the vector database
            One of ['semantic_dense', 'hybrid']
        message_render_style (str): Controls how chat messages are rendered when including them in \
model prompts.
            One of ['json_dumps', 'plain_text'].
            'plain_text' looks like "USER: ...<br>ASSISTANT: ...<br>"
            'json_dumps' gives standard openai-style chat completion messages JSON
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
        vector_memory_type: Literal[
            "chat_message", "extracted_facts"
        ] = "extracted_facts",
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
        relevant_memories: list[str] = self.fetch_relevant_memories(
            query=self.chat_messages_to_text(
                messages=self.recent_chat_messages,
                output_style=self.message_render_style,
            ),
            n_to_fetch=self.n_vector_memories_to_fetch,
            search_method=self.vector_search_method,
        )
        prompt_messages: list[ChatMessage] = (
            [ChatMessage(role="system", content=self.system_prompt)]
            # + self.recent_chat_messages
            + [self.augment_user_msg(user_msg, relevant_memories)]
        )
        logger.debug([msg.model_dump() for msg in prompt_messages])
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[msg.model_dump() for msg in self.recent_chat_messages],
        )
        assistant_response: ChatMessage = ChatMessage(
            role=llm_api_response.choices[0].message.role,
            content=llm_api_response.choices[0].message.content,
        )
        logger.debug(
            assistant_response.model_dump_json(indent=4),
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
        if len(self.recent_chat_messages) / 2 > self.n_chat_messages_in_working_memory:
            messages_to_archive: list[ChatMessage] = self.recent_chat_messages[:2]
            self.recent_chat_messages = self.recent_chat_messages[2:]
            match self.vector_memory_type:
                case "chat_message":
                    self.add_vector_memory(
                        self.chat_messages_to_text(
                            messages=messages_to_archive,
                            output_style=self.message_render_style,
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
{self.chat_messages_to_text(messages=messages, output_style=self.message_render_style)}
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
