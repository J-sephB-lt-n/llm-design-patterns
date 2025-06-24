"""
TODO: module docstring
"""

import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Final, Optional

import chonkie
import streamlit as st
import tiktoken
from pydantic import BaseModel, ConfigDict

# I absolutely hate this hack, but I want streamlit to run
#   from the project root folder and I can't find the equivalent of
#   `python -m` for `streamlit run`
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from pdf_to_text.docling_pdf_to_text import doc_to_text


def landing_page(*args, **kwargs):
    st.title("Text Chunk Algorithm Explorer")
    st.markdown(
        """
I designed this simple app in order to compare different text chunking approaches.

I'm using the implementations in the [Chonkie](https://github.com/chonkie-inc/chonkie) python package.

Use the app as follows:

1. Use **Upload Doc** page to ingest a document (currently uses [docling](https://github.com/docling-project/docling) \
for PDF to text)

2. Use **View Uploaded Doc** page to see the text of your ingested document

3. Use **Chunk Doc** page to experiment with different chunking approaches
        """.strip()
    )


def upload_doc_page(*args, **kwargs):
    st.title("Upload Doc")
    uploaded_file = st.file_uploader(
        label="Upload a document (.pdf or .txt)", type=["pdf", "txt"]
    )
    if uploaded_file:
        file_extension: str = Path(uploaded_file.name).suffix
        if file_extension == ".txt":
            try:
                st.session_state["doc_text_content"] = uploaded_file.read().decode(
                    "utf-8"
                )
            except UnicodeDecodeError:
                st.error("Failed to read file using encoding utf-8")
        elif file_extension == ".pdf":
            with st.spinner("Extracting text from PDF using docling"):
                st.session_state["doc_text_content"] = doc_to_text(
                    doc_file_extension=file_extension,
                    doc_file_content=uploaded_file.getvalue(),
                )
        else:
            raise RuntimeError("Reached a path which should be unreachable")

        st.session_state["doc_n_tokens"] = len(
            st.session_state["tokeniser"].encode(st.session_state["doc_text_content"])
        )

        st.success(f"Successfully ingested document {uploaded_file.name}")


def view_doc_page(*args, **kwargs):
    st.title("View Uploaded Document")
    if "doc_text_content" not in st.session_state:
        st.text("No document has been uploaded yet")
    else:
        st.markdown(
            f"""
Document is:
- {len(st.session_state["doc_text_content"]):,} characters long
- {len(st.session_state["doc_text_content"].split()):,} characters long
- {st.session_state["doc_n_tokens"]:,} tokens long (using tokeniser \
  '{st.session_state["tokeniser"].name}')
"""
        )
        st.text_area(
            label="Document Content",
            value=st.session_state["doc_text_content"],
            height=999,
        )


def chunk_doc_page(chunkers):
    st.title("Chunk Document")
    selected_chunker_name: str = st.selectbox(
        label="Select a Chonkie text chunker",
        options=chunkers.keys(),
    )
    chunker_def: Optional[ChunkerDef] = chunkers[selected_chunker_name]
    if chunker_def.chunker is None:
        st.warning("Not Implemented")
        st.markdown(chunker_def.explanation)
    else:
        st.markdown(chunker_def.explanation)
        user_chunker_kwargs: dict = {}

        if chunker_def.chunker is chonkie.SemanticChunker:
            threshold_mode = st.radio(
                label="threshold_mode",
                options=["cosine_similarity", "percentile", "auto"],
            )
            match threshold_mode:
                case "cosine_similarity":
                    chunker_def.streamlit_input_controls["threshold"] = partial(
                        st.number_input,
                        label="threshold: (sentence cosine similarity fixed cutoff value)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                    )
                case "percentile":
                    chunker_def.streamlit_input_controls["threshold"] = partial(
                        st.number_input,
                        label="threshold: (percentile of sentence cosine similarity scores cutoff)",
                        min_value=1,
                        max_value=99,
                        value=95,
                    )
                case "auto":
                    chunker_def.streamlit_input_controls["threshold"] = partial(
                        st.radio,
                        label="threshold = 'auto' (no further parameters here)",
                        options=[
                            "auto",
                        ],
                    )
        for argname, streamlit_control in chunker_def.streamlit_input_controls.items():
            if argname[0] != "_":
                user_chunker_kwargs[argname] = streamlit_control()

        submit_button = st.button(label="Chunk Document with Selected Chunker")

        if submit_button:
            all_chunker_kwargs: dict = (
                user_chunker_kwargs | chunker_def.hardcoded_chunker_kwargs
            )
            st.write("Final chunker params:")
            st.json(all_chunker_kwargs)
            chunker: chonkie.BaseChunker = chunker_def.chunker(**all_chunker_kwargs)
            chunks = chunker.chunk(st.session_state["doc_text_content"])
            for chunk_num, chunk in enumerate(chunks, start=1):
                st.text_area(
                    label=f"Chunk {chunk_num:,}",
                    value=chunk.text,
                    height=max(68, 35 * (chunk.text.count("\n") + 1)),
                )


def main():
    st.set_page_config(
        page_title="Chonkie Visualiser",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "tokeniser" not in st.session_state:
        st.session_state["tokeniser"] = tiktoken.get_encoding("o200k_base")

    class ChunkerDef(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        chunker: type[chonkie.BaseChunker] | Callable | None = None
        explanation: str
        streamlit_input_controls: dict[str, Callable] | None = None
        hardcoded_chunker_kwargs: dict[str, Any] | None = None

    CHUNKERS: Final[dict[str, Optional[ChunkerDef]]] = {
        "Token Chunker": ChunkerDef(
            chunker=chonkie.TokenChunker,
            explanation="""
Puts a fixed number of tokens in each chunk.
    """.strip(),
            streamlit_input_controls={
                "chunk_size": partial(
                    st.number_input,
                    label="chunk_size: (maximum number of tokens in a chunk)",
                    min_value=1,
                    value=512,
                ),
                "chunk_overlap": partial(
                    st.number_input,
                    label="chunk_overlap: (number of tokens shared/common between consecutive chunks)",
                    min_value=0,
                    value=0,
                ),
            },
            hardcoded_chunker_kwargs={"tokenizer": st.session_state["tokeniser"]},
        ),
        "Sentence Chunker": ChunkerDef(
            chunker=chonkie.SentenceChunker,
            explanation="""
Puts a fixed number of tokens in each chunk, ensuring complete sentences (won't split in the middle of a sentence)
    """.strip(),
            streamlit_input_controls={
                "chunk_size": partial(
                    st.number_input,
                    label="chunk_size: (maximum number of tokens in a chunk)",
                    min_value=1,
                    value=512,
                ),
                "chunk_overlap": partial(
                    st.number_input,
                    label="chunk_overlap: (number of tokens shared/common between consecutive chunks)",
                    min_value=0,
                    value=0,
                ),
                "min_sentences_per_chunk": partial(
                    st.number_input,
                    label="min_sentences_per_chunk: (minimum number of sentences in a chunk)",
                    min_value=1,
                    value=1,
                ),
            },
            hardcoded_chunker_kwargs={
                "tokenizer_or_token_counter": st.session_state["tokeniser"]
            },
        ),
        "Recursive Chunker": ChunkerDef(
            chunker=chonkie.RecursiveChunker.from_recipe,
            explanation="""
Continues to split chunks until they contain less than `chunk_size` tokens.

It works sequentially through a fixed list of delimiters, which by default is:

1. Chunk by paragraph ('\\n\\n')

2. Chunk by newline ('\\n')

3. Chunk by sentence ('. ', '! ', '? ')

4. Chunk by whitespace (' ')

5. Chunk by characters

| Arg                         | Description
|-----------------------------|------------------------
| chunk_size                  | Minimum number of tokens in a chunk
| min_characters_per_chunk    | Minimum number of characters in a chunk
| recipe                      | A predefined set of rules (see https://huggingface.co/datasets/chonkie-ai/recipes)
    """.strip(),
            streamlit_input_controls={
                "chunk_size": partial(
                    st.number_input,
                    label="Maximum number of tokens in a chunk",
                    min_value=1,
                    value=512,
                ),
                "min_characters_per_chunk": partial(
                    st.number_input,
                    label="Minimum number of characters in a chunk",
                    min_value=1,
                    value=1,
                ),
                "name": partial(
                    st.selectbox,
                    label="Chosen recipe (rule set)",
                    options=["default", "markdown"],
                ),
            },
            hardcoded_chunker_kwargs={
                "tokenizer_or_token_counter": st.session_state["tokeniser"],
                "lang": "en",
            },
        ),
        "Code Chunker": ChunkerDef(
            explanation="""
CodeChunker splits code using the structure (Abstract Syntax Tree) of the code.

The AST is implemented using https://github.com/tree-sitter/tree-sitter. 

This implementation supports any programming language in `tree-sitter-language-pack`
            """.strip()
        ),
        "Semantic Chunker": ChunkerDef(
            chunker=chonkie.SemanticChunker,
            explanation="""
Splits text into chunks based on semantic similarity (using dense vectors).
(i.e. semantically similar text will go into the same chunk)

The originally proposed algorithm (by Greg Kamradt) is:

1. Split the document into sentences (e.g. by splitting on ['. ', '? ', '! '])

2. Group sequential sentences together (i.e. literally concatenate the text from the sentences together)
    
3. Embed each sentence group into a semantic vector using a dense embedding model (e.g. model2vec)

4. Calculate the vector distance between consecutive vectors (e.g. group1 vs group2, group2 vs group3, group3 vs group4...)

5. Split into 2 chunks where the distance between 2 groups exceeds a chosen `threshold` value

6. Further split chunks (where necessary) to meet chunk size requirements

#### Example of behavior of `sentence_window` in the [Chonkie implementation of SemanticChunker](https://github.com/chonkie-inc/chonkie/blob/main/src/chonkie/chunker/semantic.py):

```python
sentences = ["s1", "s2", "s3", "s4", "s5"]
sentence_groups = {
    "sentence_window=1": [
        ("s1", "s2",),          # 1 neighbour on either side of "s1"
        ("s1", "s2", "s3",),    # 1 neighbour on either side of "s2"
        ("s2", "s3", "s4",),    # 1 neighbour on either side of "s3"
        ("s3", "s4", "s5",),    # 1 neighbour on either side of "s4"
        ("s4", "s5",),          # 1 neighbour on either side of "s5"
    ],
    "sentence_window=2": [
        ("s1", "s2", "s3",),                # 2 neighbours on either side of "s1"
        ("s1", "s2", "s3", "s4",),          # 2 neighbours on either side of "s2"
        ("s1", "s2", "s3", "s4", "s5",),    # 2 neighbours on either side of "s3"
        ("s2", "s3", "s4", "s5",),          # 2 neighbours on either side of "s4"
        ("s3", "s4", "s5",),                # 2 neighbours on either side of "s5"
    ],
}
```

#### Explanation of `threshold_mode` (in the [Chonkie implementation of SemanticChunker](https://github.com/chonkie-inc/chonkie/blob/main/src/chonkie/chunker/semantic.py))

- **cosine_similarity**:    Sentence group embeddings with a cosine similarity score less than a chosen fixed value \
are split into separate chunks (e.g. threshold=0.3 means a split happens for cosine similarity values below 0.3)

- **percentile**:           Sentence group embeddings with a cosine similarity score less than a chosen percentile \
of similarity scores (e.g. threshold=95 means a split happens for cosine similarity values in the bottom 5% of \
similarity values i.e. higher percentile threshold means fewer chunks)

- **auto**:                 Uses binary search to find a threshold which results in chunks of size [min_chunk_size, chunk_size]

#### Explanation of `mode` (in the [Chonkie implementation of SemanticChunker](https://github.com/chonkie-inc/chonkie/blob/main/src/chonkie/chunker/semantic.py))

- **window**:     The original implementation as described above (Greg Kamradt's method). Each sentence group (embedding) is compared \
to the sentence group (embedding) before it and a split happens if those 2 groups are different enough. i.e. a split (new chunk) happens \
when there is a big semantic jump between 2 consecutive bits of text.

- **cumulative**: Iteratively builds a chunk by keeping a rolling average embedding ("chunk centroid") upload_doc_pageross all sentence groups added to the chunk so far. \
Then, a new chunk is started (a split) when the next sentence group is different enough from the chunk centroid. A new sentence group means \
starting a new chunk centroid. i.e. a split appears when text appears which is different enough from all text in the chunk so far.

The `window` approach is better for text which changes topic frequently. The `cumulative` approach is better for text which tends to meander \
around a topic.
    """.strip(),
            streamlit_input_controls={
                "mode": partial(
                    st.selectbox,
                    label="mode: (method of grouping sentences)",
                    options=["cumulative", "window"],
                ),
                "threshold": partial(
                    # this input type can change dynamically at runtime
                    # (see 'if chunker_def.chunker is chonkie.SemanticChunker')
                    st.number_input,
                    label="threshold: (sentence similarity threshold)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                ),
                "chunk_size": partial(
                    st.number_input,
                    label="chunk_size: (maximum number of tokens in a chunk)",
                    min_value=0,
                    value=512,
                ),
                "similarity_window": partial(
                    st.number_input,
                    label="similarity_window: (number of sentences on each side to include in group - see example above)",
                    min_value=0,
                    value=1,
                ),
                "min_sentences": partial(
                    st.number_input,
                    label="min_sentences: (minimum number of sentences in a chunk)",
                    min_value=1,
                    value=1,
                ),
                "min_chunk_size": partial(
                    st.number_input,
                    label="min_chunk_size: (minimum number of tokens in a chunk)",
                    min_value=1,
                    value=2,
                ),
                "min_characters_per_sentence": partial(
                    st.number_input,
                    label="min_characters_per_sentence: (minimum number of characters in a sentence)",
                    min_value=1,
                    value=12,
                ),
                "threshold_step": partial(
                    st.number_input,
                    label="threshold_step: (controls granularity of binary search when using threshold='auto')",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.01,
                ),
            },
            hardcoded_chunker_kwargs={
                "embedding_model": "minishlab/potion-base-8M",
            },
        ),
        "SDPM Chunker": ChunkerDef(
            explanation="""
The SDPMChunker extends SemanticChunker by using a double-pass merging approach. 

How it works is to perform standard semantic chunking (SemanticChunker in this app) and then to afterward perform a \
chunk merging step, in which nearby chunks which are semantically similar but have dissimilar chunk(s) inbetween them \
can be combined into a single chunk.

This is useful in cases in which semantically very different but still very relevant text appears within a body of \
semantically-related text. An example is a mathematical formula within a paragraph of text.

There is a very good explanation of the approach [here](https://bitpeak.com/chunking-methods-in-rag-methods-comparison/)
            """.strip()
        ),
        "Late Chunker": ChunkerDef(
            explanation="""
The LateChunker implements the late chunking strategy described in the \
[Late Chunking](https://arxiv.org/abs/2409.04701) paper.

It builds on top of the RecursiveChunker and uses document-level embeddings to create more \
semantically rich chunk representations.

Instead of generating embeddings for each chunk independently, the LateChunker first encodes \
the entire text into a single embedding. It then splits the text using recursive rules and \
derives each chunk’s embedding by averaging relevant parts of the full document embedding. \
This allows each chunk to carry broader contextual information, improving retrieval performance \
in RAG systems.
    """.strip()
        ),
        "Cache-Augmented Generation (CAG)": ChunkerDef(
            explanation="""
CAG is not implemented in Chonkie but I really wanted to make a note of it here.

CAG stores the entire knowledge base in the model context window, but precomputes the attention layer key/value \
calculation so that it doesn't need to be recomputed for every user query.

Here is the original CAG paper: https://arxiv.org/abs/2412.15605v1
            """,
        ),
        "Neural Chunker": ChunkerDef(
            explanation="""
The NeuralChunker uses a fine-tuned BERT model specifically trained to identify semantic shifts \
within text, allowing it to split documents at points where the topic or context changes \
significantly. This provides highly coherent chunks ideal for RAG.

The default model is [mirth/chonky_modernbert_base_1](https://huggingface.co/mirth/chonky_modernbert_base_1)
    """.strip()
        ),
        "Slumber Chunker": ChunkerDef(
            explanation="""
This approach uses a large language model to decide how text should be chunked.

I do not think that the implementation in this package (Chonkie) is particularly strong, and \
I would implement this myself from first principles in practice.

For example, Chonkie makes no use of structured outputs (e.g. https://github.com/567-labs/instructor) \
to guarantee that the model returns JSON in the correct format, which is nowadays standard \
practice.

Here is the prompt which Chonkie uses:

```
<task> You are given a set of texts between the starting tag <passages> and ending tag </passages>.

Each text is labeled as 'ID `N`' where 'N' is the passage number. 

Your task is to find the first passage where the content clearly separates from the previous \
passages in topic and/or semantics. </task>

<rules>
Follow the following rules while finding the splitting passage:
- Always return the answer as a JSON parsable object with the 'split_index' key having a value \
of the first passage where the topic changes.
- Avoid very long groups of paragraphs. Aim for a good balance between identifying content \
shifts and keeping groups manageable.
- If no clear `split_index` is found, return N + 1, where N is the index of the last passage. 
</rules>

<passages>
{passages}
</passages>
```
            """.strip()
        ),
    }

    PAGES: Final[dict[str, Callable]] = {
        "Landing": landing_page,
        "Upload Doc": upload_doc_page,
        "View Uploaded Doc": view_doc_page,
        "Chunk Doc": chunk_doc_page,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page(chunkers=CHUNKERS)


if __name__ == "__main__":
    main()
