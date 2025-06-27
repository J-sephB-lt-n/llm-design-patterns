
# ChatBot Memory Explorer

Plan:

- A streamlit web app where you can try out different agent memory approaches (and hyperparameters) using a chat interface

- A setup screen for LLM setup

- A description of each approach (so can use the app to learn)

- Ability to inspect the memory store

- Ability to expand/collapse individual chat messages i.e. for a single turn toggle showing just the user input message and agent response, or the full chain of messages (system prompt + user prompt with all automatically added content + tool calls + tool results + raw agent response + parsed agent response)

- algs all follow a fixed protocol

- can add new algs using a decorator
