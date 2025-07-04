import streamlit as st

st.session_state.messages = [
    {
        "role": "system",
        "content": """
        a 
system 
    message""",
    },
    {
        "role": "user",
        "content": "a user message",
    },
    {
        "role": "assistant",
        "content": "the agent response",
    },
]

with st.expander("Show chat detail", expanded=False):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
