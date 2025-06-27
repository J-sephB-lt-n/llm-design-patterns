
# ChatBot Memory Explorer

Plan:

- A streamlit web app where you can try out different agent memory approaches (and hyperparameters) using a chat interface

- A setup screen for LLM setup

- A description of each approach (so can use the app to learn)

- Ability to inspect the memory store

- Ability to expand/collapse individual chat messages i.e. for a single turn toggle showing just the user input message and agent response, or the full chain of messages (system prompt + user prompt with all automatically added content + tool calls + tool results + raw agent response + parsed agent response)

- algs all follow a fixed protocol

- can add new algs using a decorator

Project structure:
```bash
.
├── 📁 app/                         
│   │
│   ├── 📁 memory/                  # Different memory algorithms (e.g. buffer, vector, etc.)
│   │   ├── alg1.py         
│   │   ├── alg2.py         
│   │   └── ...             
│   │
│   ├── 📁 interfaces/             # Interfaces/protocols/abstract classes
│   │   ├── memory_protocol.py     
│   │   ├── vector_db_protocol.py           
│   │   └── ...
│   │
│   ├── 📁 db/                    # Specific database implementation
│   │   ├── lancedb.py           
│   │   └── ...
│   │
│   ├── 📁 services/               # Business logic
│   │   └── ...
│   │
│   ├── 📁 utils/                  # Helper functions/utilities (e.g. logging)
│   │   └── ...
│   │
├── 📁 temp_storage/              # For temporary local files (e.g., SQLite .db, logs)
│   │   └── ...
│   │
├── 📁 tests/                     # Unit and integration tests
│   └── ...
│
├── 📄 main.py                    # Entrypoint for Streamlit (`streamlit run main.py`)
```
```
