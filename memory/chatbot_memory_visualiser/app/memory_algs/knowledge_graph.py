"""
Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database
"""

import itertools
import json
import re
import unicodedata
from collections import deque
from pathlib import Path
from typing import Final, Literal, NamedTuple

import lancedb
import model2vec
import networkx as nx
import openai
import pyarrow as pa
from lancedb.rerankers import RRFReranker
from loguru import logger

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg

LLM_SYSTEM_PROMPT: Final[str] = """
You are a creative assistant who is having a conversation with a user
""".strip()

LLM_RESPOND_PROMPT: Final[str] = """
<potentially-relevant-information>
{retrieved_knowledge}
</potentially-relevant-information>

<recent-chat-history>
{recent_chat_history}
</recent-chat-history>

<latest-user-message>
{latest_use_message}
</user-user-message>

By referring to your most recent interaction with the user ("recent chat history") and the \
potentially relevant information retrieved from the long-term chat history (if relevant), \
respond the latest user message.
""".strip()

LLM_EXTRACT_KNOWLEDGE_TRIPLES_PROMPT: Final[str] = """
<conversation-snippet>
{conversation_snippet}
</conversation-snippet>

From the provided conversation snippet between a user and an assistant, extract all information \
which will be relevant to future interactions, in the form of a list of fact triples.

Each fact must be associated with one or both of the personas 'user' and/or 'assistant'.

<required-output-format>
Your response must include a JSON markdown codeblock containing a single list of lists, where \
each inner list contains exactly 3 strings (subject, predicate, object):
```json
[
    ["subject", "predicate", "object"],
    [...],
    ...
]
```
Include spaces between words in subject, predicate and object.
</required-output-format>
""".strip()

LLM_RDF_TRIPLES_DEDUP_PROMPT: Final[str] = """
<proposed-new-knowledge-triples>
```json
{new_rdf_triples}
```
</proposed-new-knowledge-triples>

You have been provided with a list of new knowledge (RDF) triples proposed to be added to \
the existing knowledge database. For each proposed new triple, the closest {n_closest_triples} \
existing knowledge triples in the database are shown.

Your task is as follows:

1. Decide which of the proposed new triples should be added to the database - return just the \
list of triples to add. Omit triples whose knowledge content is already in the database (i.e. \
omit those that do not add any new information).
2. If one of the new proposed triples contains a formatting inconsistency with one of the \
triples in the existing knowledge database (e.g. the same subject/object is referenced but \
with a difference in punctuation or case), then correct it in your list of triples to add.

<required-output-format>
Your response must include a JSON markdown codeblock containing a single list of lists, where \
each inner list contains exactly 3 strings (subject, predicate, object):
```json
[
    ["subject", "predicate", "object"],
    [...],
    ...
]
```
</required-output-format>
"""


class KnowledgeTriple(NamedTuple):
    subj: str
    pred: str
    obj: str


class KnowledgeGraphMemory(MemoryAlg):
    """
    Memory algorithm which stores memories as a knowledge graph

    More specifically, the algorithm works as follows:
        - At each new user message in the chat, a LLM is used to extract semantic (RDF) triples \
from the last `triples_source_n_msgs` chat messages in the conversation
        - The subject, predicate and object text content are cleaned, to prevent duplication \
in the graph (strip, lowercase etc.)
        - These new triples are deduped against the existing knowledge triples already in the \
graph using a LLM to make the decisions
        - Triples which are new (i.e. not duplicates) are added to the graph - the subject and \
object as nodes and the predicate as an edge.
        - The content of the new nodes (subject and object) is embedded and added into a vector \
database (for later lookup)
        - Each new user chat message is processed as follows:
            - The last `recent_chat_history_n_messages` (including the new user message) are \
              included in the prompt
            - The nearest `n_context_nodes` nodes whose content is most similar to the combined \
content of the last `context_n_messages` most recent chat messages are retrieved.
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
        context_n_messages (int): Number of most recent chat messages to use as query when \
fetching the most relevant nodes from the graph 
        n_context_hops (int): Number of steps to take when traversing the graph to find \
possibly related knowledge, outward from the first `n_context_nodes` nodes found.
        vector_search_method (str): Approach used for fetching nodes from the vector database
            One of ['semantic_dense', 'hybrid']
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
        system_prompt: str = LLM_SYSTEM_PROMPT,
        recent_chat_history_n_messages: int = 10,
        triples_source_n_messages: int = 4,
        n_context_nodes: int = 3,
        context_n_messages: int = 6,
        n_context_hops: int = 1,
        vector_search_method: Literal["semantic_dense", "hybrid"] = "hybrid",
        message_render_style: Literal["plain_text", "json_dumps", "xml"] = "plain_text",
    ) -> None:
        if recent_chat_history_n_messages < context_n_messages:
            raise ValueError(
                "`context_n_messages` cannot exceed number of messages in recent chat "
                "history (`recent_chat_history_n_messages`)"
            )
        self.chat_history: list[ChatMessageDetail] = []
        self.recent_chat_messages: deque[ChatMessage] = deque(
            maxlen=recent_chat_history_n_messages
        )

        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.system_prompt = system_prompt
        self.recent_chat_history_n_messages = recent_chat_history_n_messages
        self.triples_source_n_messages = triples_source_n_messages
        self.n_context_nodes = n_context_nodes
        self.context_n_messages = context_n_messages
        self.n_context_hops = n_context_hops
        self.vector_search_method = vector_search_method
        self.message_render_style = message_render_style

        self.graph = nx.MultiDiGraph()

        self.embed_model = model2vec.StaticModel.from_pretrained(
            "minishlab/potion-retrieval-32M"
        )

        # set up temporary vector database on filesystem #
        self.vector_db = lancedb.connect(Path("temp_files/graph_memory_nodes_db"))
        self.node_embeddings = self.vector_db.create_table(
            name="graph_nodes",
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
            self.node_embeddings.create_fts_index("text")
            self.reranker = RRFReranker()

    def normalise_text(self, text: str) -> str:
        """
        Simplifies text as far as possible, including diacritic removal
        """
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ASCII", "ignore").decode("ASCII")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        text = text.lower().strip()

        return text

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
            case _:
                raise ValueError(f"Unknown output style '{message_render_style}'")

    def add_knowledge_triple(self, new_triple: KnowledgeTriple) -> None:
        """
        Adds knowledge triple into the knowledge graph and subject and predicate as \
nodes in the vector database
        """
        subj = new_triple.subj
        pred = new_triple.pred
        obj = new_triple.obj
        for node_id in (subj, obj):
            if node_id not in self.graph:
                self.graph.add_node(node_id)
        if self.graph.has_edge(subj, obj, key=pred):
            self.graph[subj][obj][pred]["weight"] += 1
        else:
            self.graph.add_edge(
                subj,
                obj,
                key=pred,
                weight=1,
            )

        # self.node_embeddings.add(
        #     [
        #         {
        #             "text": "TODO",
        #             "node_id": "TODO",
        #             "vector": self.embed_model.encode("TODO"),
        #         }
        #     ]
        # )

    # def fetch_relevant_knowledge_triples(
    #     self,
    #     query: str,
    # ) -> list[KnowledgeTriple]:
    #     """
    #     Fetch existing semantic triples closest to `query`
    #     """
    #     embed_query: np.ndarray = self.embed_model.encode(query)
    #     match search_method:
    #         case "semantic_dense":
    #             search_results = (
    #                 self.vector_memories.search(embed_query).limit(n_to_fetch).to_list()
    #             )
    #         case "hybrid":
    #             search_results = (
    #                 self.vector_memories.search(query_type="hybrid")
    #                 .vector(embed_query)
    #                 .text(query)
    #                 .rerank(self.reranker)
    #                 .limit(n_to_fetch)
    #                 .to_list()
    #             )
    #         case _:
    #             raise ValueError(f"Unknown search_method='{search_method}'")
    #
    #     return [x["text"] for x in search_results]

    def extract_knowledge_triples(
        self,
        source_messages: list[ChatMessage],
        message_render_style: Literal["plain_text", "json_dumps", "xml"],
    ) -> list[KnowledgeTriple]:
        """
        Extract all facts in `chat_messages` as a list of `KnowledgeTriple`s
        """
        prompt_messages: list[ChatMessage] = [
            ChatMessage(
                role="user",
                content=LLM_EXTRACT_KNOWLEDGE_TRIPLES_PROMPT.format(
                    conversation_snippet=self.chat_messages_to_text(
                        messages=source_messages,
                        message_render_style=message_render_style,
                    )
                ),
            )
        ]
        logger.debug([msg.model_dump() for msg in prompt_messages])
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[msg.model_dump() for msg in prompt_messages],
        )
        logger.debug(
            {
                "role": llm_api_response.choices[0].message.role,
                "content": llm_api_response.choices[0].message.content,
            }
        )
        json_codeblock: str = re.search(
            r"```json\s*\n(?P<json_content>.*?)\n```",
            llm_api_response.choices[0].message.content,
            re.DOTALL,
        ).group("json_content")
        raw_triples: list[list[str]] = json.loads(json_codeblock)
        triples: list[KnowledgeTriple] = [
            KnowledgeTriple(
                subj=self.normalise_text(subj),
                pred=self.normalise_text(pred),
                obj=self.normalise_text(obj),
            )
            for subj, pred, obj in raw_triples
        ]
        return triples

    def dedup_knowledge_triples(
        self, new_triples: list[KnowledgeTriple]
    ) -> list[KnowledgeTriple]:
        """
        Remove `new_triples` with knowledge already in the knowledge base
        """
        return new_triples

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
        # relevant_knowledge_tuples: list["TODO"] = self.fetch_relevant_nodes(
        #     # query=self.chat_messages_to_text(
        #     #     messages=self.recent_chat_messages,
        #     #     output_style=self.message_render_style,
        #     # ),
        #     # n_to_fetch=self.n_vector_memories_to_fetch,
        #     # search_method=self.vector_search_method,
        # )
        prompt_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            # ChatMessage(role="user", content="TODO"),
        ]
        # >>> TEMP DEV (start) <<< #
        prompt_messages += [msg for msg in self.recent_chat_messages]
        # >>> TEMP DEV (end) <<< #
        logger.debug([msg.model_dump() for msg in prompt_messages])
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
            assistant_response.model_dump_json(indent=4),
        )
        self.recent_chat_messages.append(assistant_response)

        proposed_new_rdf_triples: list[KnowledgeTriple] = (
            self.extract_knowledge_triples(
                source_messages=list(self.recent_chat_messages)[
                    -self.triples_source_n_messages :
                ],
                message_render_style=self.message_render_style,
            )
        )
        logger.debug(f"proposed new knowledge triples: {proposed_new_rdf_triples}")

        triples_to_add: list[KnowledgeTriple] = self.dedup_knowledge_triples(
            proposed_new_rdf_triples
        )

        for triple_to_add in triples_to_add:
            self.add_knowledge_triple(triple_to_add)

        # self.chat_history.append(
        #     ChatMessageDetail(
        #         visible_messages=[
        #             ChatMessage(role="user", content=user_msg),
        #             assistant_response,
        #         ],
        #         all_messages=prompt_messages + [assistant_response],
        #         token_usage=llm_api_response.usage.model_dump(),
        #     )
        # )
        # if len(self.recent_chat_messages) / 2 > self.n_chat_messages_in_working_memory:
        #     messages_to_archive: list[ChatMessage] = self.recent_chat_messages[:2]
        #     self.recent_chat_messages = self.recent_chat_messages[2:]
        #     match self.vector_memory_type:
        #         case "chat_message":
        #             self.add_vector_memory(
        #                 self.chat_messages_to_text(
        #                     messages=messages_to_archive,
        #                     output_style=self.message_render_style,
        #                 )
        #             )
        #         case "extracted_facts":
        #             facts: list[str] = self.extract_facts_from_chat_snippet(
        #                 messages_to_archive
        #             )
        #             for fact in facts:
        #                 self.add_vector_memory(fact)

    def view_memory_as_json(self) -> dict:
        """
        Render latest state of the agent's memory as a dict
        """
        return {
            "recent_chat_history": [
                msg.model_dump() for msg in self.recent_chat_messages
            ],
            "knowledge_triples": [
                (u, k, v, d)
                for u, v, k, d in self.graph.edges(
                    keys=True,
                    data=True,
                )
            ],
            # "long_term_chat_history": [
            #     x["text"] for x in self.vector_memories.search().to_list()
            # ],
        }
