"""
Example of FAST local embedding model (model2vec) with FAST local vector database (lancedb)
"""

import shutil
from pathlib import Path
from typing import Final

import lancedb
from lancedb.rerankers import RRFReranker
import model2vec
import numpy as np

DB_PATH: Final[Path] = Path("rag/temp_vector_db")

docs: list[str] = [
    "In my team, we all use python",
    "The python has no venom",
    "monty python's flying circus",
]


embedder = model2vec.StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
embed_docs: np.ndarray = embedder.encode(docs)

db = lancedb.connect(DB_PATH)

# create table in vector database and insert embeddings #
db.create_table(
    name="docs_table",
    data=[
        {
            "text": doc,
            "vector": vec,
        }
        for doc, vec in zip(docs, embed_docs)
    ],
)

docs_table = db.open_table("docs_table")

# create index for Full-Text-Search #
docs_table.create_fts_index("text")
docs_table.wait_for_index(["text_idx"])

queries = [
    "entertainment",
    "technology",
    "jungle",
]

reranker = RRFReranker()

for query in queries:
    embed_query: np.ndarray = embedder.encode(query)
    search_results = docs_table.search(embed_query).limit(1).to_list()
    print(
        f"""closest doc to query '{query}' using semantic (vector) search is '{search_results[0]["text"]}'"""
    )

    hybrid_search_results = (
        docs_table.search(query_type="hybrid")
        .vector(embed_query)
        .text(query)
        .limit(1)
        .to_list()
    )
    print(
        f"""closest doc to query '{query}' using hybrid search is '{hybrid_search_results[0]["text"]}'"""
    )

# delete the database #
shutil.rmtree(DB_PATH)
