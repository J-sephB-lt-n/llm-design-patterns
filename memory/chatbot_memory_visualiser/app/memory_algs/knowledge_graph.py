"""
Memory algorithm which stores the few most recent chat messages and the full chat history \
in a vector database
"""

import json
import re
from collections import deque
from pathlib import Path
from typing import Final, Literal, NamedTuple

import lancedb
import model2vec
import networkx as nx
import numpy as np
import openai
import plotly.graph_objects as go
import pyarrow as pa
import streamlit as st
from lancedb.rerankers import RRFReranker
from loguru import logger

from app.lifecycle.teardown import app_cleanup
from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg

LLM_SYSTEM_PROMPT: Final[str] = (
    """
You are a creative assistant who is having a conversation with a user
""".strip()
)

LLM_RESPOND_PROMPT: Final[str] = (
    """
<long-term-chat-history>
{long_term_chat_history}
</long-term-chat-history>

<recent-chat-history>
{recent_chat_history}
</recent-chat-history>

<current-user-message>
{current_user_message}
</current-user-message>

By referring to your most recent interactions with the user ("recent chat history") and \
the retrieved memories from long-term chat history represented as knowledge triples (if \
they are relevant), respond to the current user message.
""".strip()
)

LLM_EXTRACT_KNOWLEDGE_TRIPLES_PROMPT: Final[str] = (
    """
<conversation-snippet>
{conversation_snippet}
</conversation-snippet>

From the provided conversation snippet between a user and an assistant, extract all information \
which will be relevant to future interactions, in the form of a list of fact triples.

Where a subject or object is associated with either the user or the assistant, then include this \
in the subject/object name (e.g. "user's wife", "assistant's opinion").

Bear in mind that we are aiming to build a wide knowledge graph which will be traversed in order \
to retrieve relevant parts of the conversation. If every extracted triple has "assistant" and \
"user" as subject and object then the graph will have 2 nodes with many edges, which is not \
useful.

<required-output-format>
Your response must include a JSON markdown codeblock containing a single list of lists, where \
each inner list contains exactly 3 strings (subject, predicate, object):
```json
[
    ["subject", "predicate", "object"],
    ...
]
```
Include spaces between words in subject, predicate and object.
</required-output-format>
""".strip()
)

LLM_RDF_TRIPLES_DEDUP_PROMPT: Final[
    str
] = """
<proposed-new-knowledge-triples>
```
{new_rdf_triples}
```
</proposed-new-knowledge-triples>

<existing-triples>
{existing_triples}
</existing-triples>

You have been provided with a list of new knowledge (RDF) triples proposed to be added to \
the existing knowledge database, as well as the most similar existing knowledge triples \
already in the database.

Your task is as follows:

1. Return the list of new triples which should be added to the database - return just the \
list of triples to add. Omit triples whose knowledge content is already in the database (i.e. \
omit those that do not add any new information).
2. If one of the new proposed triples contains a formatting inconsistency with one of the \
triples in the existing knowledge database (e.g. the same subject/object is referenced but \
with a difference in punctuation or case), then format the new triple so that it is \
consistent with the existing triples.
3. If there is any inconsistency amongst the proposed new triples (e.g. the same subject \
or object is referenced in two different triples but with different case in each) then \
make them consistent.

<required-output-format>
Your response must include a JSON markdown codeblock containing a single list of lists, where \
each inner list contains exactly 3 strings (subject, predicate, object):
```json
[
    ["subject", "predicate", "object"],
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
        - These new triples are deduped against the existing knowledge triples already in the \
graph using a LLM to make the decisions
        - Triples which are new (i.e. not duplicates) are added to the graph - the subject and \
object as nodes and the predicate as an edge.
        - The new triples are embedded and added into a vector database (for later lookup)
        - Each new user chat message is processed as follows:
            - The last `recent_chat_history_n_messages` (including the new user message) are \
              included in the prompt
            - The nearest `n_context_triples` embedded triples whose content is most similar \
              to the combined content of the last `context_n_messages` most recent chat \
              messages are retrieved.
            - For each of these retrieved triples, the graph is traversed by `n_context_hops` \
steps outward from the subject and object nodes of that triple, and all of the knowledge \
triples explored in this way are included as context in the prompt.

    Attributes:
        llm_client (openai.OpenAI): Model API client
        llm_name (str): Model name (model identifier in model API)
        llm_temperature (float): Model temperature (level of model output randomness)
        system_prompt (str): Preliminary instructions given to the language model
        recent_chat_history_n_messages (int): Number of most recent chat messages to include \
directly in the prompt
        triples_source_n_messages (int): Number of most recent chat messages to use to extract \
knowledge triples
        n_context_triples (int): Number of initial triples fetched from the knowledge graph \
to add as context to the prompt
        context_n_messages (int): Number of most recent chat messages to use as query when \
fetching the most relevant knowledge triples from the graph 
        n_context_hops (int): Number of steps to take when traversing the graph to find \
possibly related knowledge, outward from the first `n_context_nodes` nodes found.
        dedup_n_comparison_triples (int): Number of most similar triples to fetch (per \
newly proposed triple) in order to check for duplicated knowledge.
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
        recent_chat_history_n_messages: int = 8,
        triples_source_n_messages: int = 4,
        n_context_triples: int = 3,
        context_n_messages: int = 2,
        n_context_hops: int = 2,
        dedup_n_comparison_triples: int = 20,
        vector_search_method: Literal["semantic_dense", "hybrid"] = "hybrid",
        message_render_style: Literal["plain_text", "json_dumps", "xml"] = "plain_text",
    ) -> None:
        # delete any temporary files created by previous alg #
        app_cleanup()

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
        self.n_context_triples = n_context_triples
        self.context_n_messages = context_n_messages
        self.n_context_hops = n_context_hops
        self.dedup_n_comparison_triples = dedup_n_comparison_triples
        self.vector_search_method = vector_search_method
        self.message_render_style = message_render_style

        self.graph = nx.MultiDiGraph()

        self.embed_model = model2vec.StaticModel.from_pretrained(
            "minishlab/potion-retrieval-32M"
        )

        # set up temporary vector database on filesystem #
        self.vector_db = lancedb.connect(Path("temp_files/graph_memory_nodes_db"))
        self.knowledge_triple_embeddings = self.vector_db.create_table(
            name="graph_nodes",
            schema=pa.schema(
                [
                    pa.field("text", pa.string()),
                    pa.field("subject", pa.string()),
                    pa.field("predicate", pa.string()),
                    pa.field("object", pa.string()),
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
            self.knowledge_triple_embeddings.create_fts_index("text")
            self.reranker = RRFReranker()

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

    def add_knowledge_triple(self, new_triple: KnowledgeTriple) -> None:
        """
        Add new knowledge triple into the knowledge graph and vector database
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

        self.knowledge_triple_embeddings.add(
            [
                {
                    "text": f"{subj} {pred} {obj}",
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "vector": self.embed_model.encode(f"{subj} {pred} {obj}"),
                }
            ]
        )

    def fetch_relevant_knowledge_triples(
        self,
        query: str,
        search_method: Literal["semantic_dense", "hybrid"],
        n_to_fetch: int,
    ) -> list[KnowledgeTriple]:
        """
        Fetch `n_to_fetch` existing semantic triples closest to `query` using `search_method`
        """
        embed_query: np.ndarray = self.embed_model.encode(query)
        match search_method:
            case "semantic_dense":
                search_results = (
                    self.knowledge_triple_embeddings.search(embed_query)
                    .limit(n_to_fetch)
                    .to_list()
                )
            case "hybrid":
                search_results = (
                    self.knowledge_triple_embeddings.search(query_type="hybrid")
                    .vector(embed_query)
                    .text(query)
                    .rerank(self.reranker)
                    .limit(n_to_fetch)
                    .to_list()
                )

        logger.debug(
            f"""
            Fetched relevant knowledge triples: 
Query: "{query}"
Results:
{"\n".join("\t" + x["text"] for x in search_results)}
            """.strip(),
        )
        return [
            KnowledgeTriple(subj=x["subject"], pred=x["predicate"], obj=x["object"])
            for x in search_results
        ]

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
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[msg.model_dump() for msg in prompt_messages],
        )
        assistant_response: ChatMessage = ChatMessage(
            role=llm_api_response.choices[0].message.role,
            content=llm_api_response.choices[0].message.content,
        )
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[],
                all_messages=[*prompt_messages, assistant_response],
                token_usage=llm_api_response.usage.model_dump(),
            )
        )
        json_codeblock: str = re.search(
            r"```json\s*\n(?P<json_content>.*?)\n```",
            assistant_response.content,
            re.DOTALL,
        ).group("json_content")
        raw_triples: list[list[str]] = json.loads(json_codeblock)
        triples: list[KnowledgeTriple] = [
            KnowledgeTriple(
                subj=subj.strip(),  # self.normalise_text(subj),
                pred=pred.strip(),  # self.normalise_text(pred),
                obj=obj.strip(),  # self.normalise_text(obj),
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
        existing_triples: set[KnowledgeTriple] = set()
        for new_triple in new_triples:
            for similar_triple in self.fetch_relevant_knowledge_triples(
                query=" ".join(new_triple),
                n_to_fetch=self.dedup_n_comparison_triples,
                search_method=self.vector_search_method,
            ):
                existing_triples.add(similar_triple)
        prompt_messages: list[ChatMessage] = [
            ChatMessage(
                role="user",
                content=LLM_RDF_TRIPLES_DEDUP_PROMPT.format(
                    new_rdf_triples="\n".join(
                        str(list(triple)) for triple in new_triples
                    ),
                    existing_triples="\n".join(
                        str(list(triple)) for triple in existing_triples
                    ),
                ),
            )
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
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[],
                all_messages=[
                    *prompt_messages,
                    assistant_response,
                ],
                token_usage=llm_api_response.usage.model_dump(),
            )
        )
        json_codeblock: str = re.search(
            r"```json\s*\n(?P<json_content>.*?)\n```",
            assistant_response.content,
            re.DOTALL,
        ).group("json_content")
        raw_triples: list[list[str]] = json.loads(json_codeblock)
        deduped_triples: list[KnowledgeTriple] = [
            KnowledgeTriple(
                subj=subj.strip(),  # self.normalise_text(subj),
                pred=pred.strip(),  # self.normalise_text(pred),
                obj=obj.strip(),  # self.normalise_text(obj),
            )
            for subj, pred, obj in raw_triples
        ]
        return deduped_triples

    def expand_graph_neighbourhood(
        self,
        start_triples: list[KnowledgeTriple],
        n_hops: int,
    ) -> list[KnowledgeTriple]:
        """
        Traverse the knowledge graph outward from the nodes in `start_triples` by n_hops` steps \
and return all unique knowledge triples (including `start_triples`) discovered along the way.

        Triples containing the node(s) "user" and/or "assistant" are included, but these nodes \
are not traversed from (since these nodes occur in too many of the knowledge triples)
        """
        discovered_triples: set[KnowledgeTriple] = set(start_triples)
        nodes_to_expand: set[str] = {
            node
            for triple in start_triples
            for node in (triple.subj, triple.obj)
            if node.lower() not in ("user", "assistant")
        }

        visited_nodes: set[str] = set()
        for _ in range(n_hops):
            if not nodes_to_expand:
                break
            next_nodes_to_expand: set[str] = set()
            for node in nodes_to_expand:
                visited_nodes.add(node)

                # traverse outwards from current node
                for src_node, dest_node, edge in self.graph.out_edges(node, keys=True):
                    discovered_triples.add(
                        KnowledgeTriple(
                            subj=src_node,
                            pred=edge,
                            obj=dest_node,
                        )
                    )
                    if dest_node.lower() not in ("user", "assistant"):
                        next_nodes_to_expand.add(dest_node)

                for src_node, dest_node, edge in self.graph.in_edges(node, keys=True):
                    discovered_triples.add(
                        KnowledgeTriple(
                            subj=src_node,
                            pred=edge,
                            obj=dest_node,
                        )
                    )
                    if src_node.lower() not in ("user", "assistant"):
                        next_nodes_to_expand.add(src_node)

            nodes_to_expand = next_nodes_to_expand - visited_nodes

        return list(discovered_triples)

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
        initial_relevant_knowledge_triples: list[KnowledgeTriple] = (
            self.fetch_relevant_knowledge_triples(
                # query=self.chat_messages_to_text(
                #     messages=list(self.recent_chat_messages)[-self.n_context_triples :],
                #     message_render_style=self.message_render_style,
                # ),
                query="\n".join(
                    msg.content
                    for msg in list(self.recent_chat_messages)[
                        -self.n_context_triples :
                    ]
                ),
                n_to_fetch=self.n_context_triples,
                search_method=self.vector_search_method,
            )
        )
        logger.debug(f"Initial relevant triples: {initial_relevant_knowledge_triples}")
        relevant_knowledge_triples: list[KnowledgeTriple] = (
            self.expand_graph_neighbourhood(
                start_triples=initial_relevant_knowledge_triples,
                n_hops=self.n_context_hops,
            )
        )
        logger.debug(f"Expanded relevant triples: {relevant_knowledge_triples}")
        prompt_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=LLM_RESPOND_PROMPT.format(
                    long_term_chat_history="\n".join(
                        str(list(triple)) for triple in relevant_knowledge_triples
                    ),
                    recent_chat_history=self.chat_messages_to_text(
                        messages=list(self.recent_chat_messages)[:-1],
                        message_render_style=self.message_render_style,
                    ),
                    current_user_message=self.chat_messages_to_text(
                        messages=[
                            ChatMessage(
                                role="user",
                                content=user_msg,
                            )
                        ],
                        message_render_style=self.message_render_style,
                    ),
                ),
            ),
        ]
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
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[
                    ChatMessage(role="user", content=user_msg),
                    assistant_response,
                ],
                all_messages=[
                    *prompt_messages,
                    assistant_response,
                ],
                token_usage=llm_api_response.usage.model_dump(),
            )
        )

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
            "vectors": [x for x in self.knowledge_triple_embeddings.search().to_list()],
        }

    def custom_streamlit_plot(self) -> None:
        """
        Show a basic visualisation of the knowledge graph in the streamlit app
        Note: This code written by Gemini 2.5 pro and I haven't cleaned it up yet
        """
        G = self.graph
        if not G.nodes:
            st.info("Knowledge graph is empty.")
            return

        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=500)

        # Create edge trace
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node trace
        node_x, node_y, node_text, node_deg = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_deg.append(G.degree(node))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=[10 + 2 * d for d in node_deg],
                line=dict(width=2),
                color="lightblue",
            ),
            # The 'hovermarker' attribute is not a standard property. Highlighting on hover
            # for individual points in a trace without callbacks is a known limitation.
            # The change in hovermode will make elements more responsive, though.
        )

        # Create edge label trace
        edge_label_x, edge_label_y, edge_label_text = [], [], []
        for u, v, key in G.edges(keys=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_label_x.append((x0 + x1) / 2)
            edge_label_y.append((y0 + y1) / 2)
            edge_label_text.append(str(key))

        edge_label_trace = go.Scatter(
            x=edge_label_x,
            y=edge_label_y,
            mode="text",
            text=edge_label_text,
            hoverinfo="none",  # Labels themselves don't need hover info
            textfont=dict(size=10),
        )

        # Create a trace for invisible markers on edges for hovering
        edge_hover_x, edge_hover_y, edge_hover_text = [], [], []
        for u, v, key in G.edges(keys=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_hover_x.append((x0 + x1) / 2)
            edge_hover_y.append((y0 + y1) / 2)
            edge_hover_text.append(f"subject: {u}<br>predicate: {key}<br>object: {v}")

        edge_hover_trace = go.Scatter(
            x=edge_hover_x,
            y=edge_hover_y,
            mode="markers",
            hoverinfo="text",
            text=edge_hover_text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=15, color="rgba(0,0,0,0)"),  # Invisible markers
            hoverlabel=dict(bgcolor="#444444", font=dict(color="white")),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace, edge_label_trace, edge_hover_trace]
        )

        # Add arrows
        annotations = [
            dict(
                ax=pos[edge[0]][0],
                ay=pos[edge[0]][1],
                axref="x",
                ayref="y",
                x=pos[edge[1]][0],
                y=pos[edge[1]][1],
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#888",
            )
            for edge in G.edges()
        ]

        fig.update_layout(
            showlegend=False,
            hovermode="closest",  # Enable closest hover
            margin=dict(b=10, l=5, r=5, t=10),
            annotations=annotations,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

        # Keep aspect ratio square for prettier layout
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig, use_container_width=True)
