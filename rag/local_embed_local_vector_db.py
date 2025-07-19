"""
Example of FAST local embedding model (model2vec) with FAST local vector database (lancedb)
"""

import shutil
from pathlib import Path
from typing import Final

import lancedb
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

queries = [
    "entertainment",
    "technology",
    "jungle",
]

for query in queries:
    embed_query: np.ndarray = embedder.encode(query)
    search_results = docs_table.search(embed_query).limit(1).to_list()
    print(f"""closest doc to query '{query}' is '{search_results[0]["text"]}'""")

# delete the database #
shutil.rmtree(DB_PATH)
