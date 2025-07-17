"""
Memory algorithm with no memory (baseline)
"""

from textwrap import dedent

import openai

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


class NoMemory(MemoryAlg):
    """
    Memory algorithm with no memory (baseline)
    """

    alg_description = dedent(
        """
        No memory whatsoever.
        """
    ).strip()

    def __init__(
        self,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.llm_client = llm_client
        self.chat_history: list[ChatMessageDetail] = []
        self.llm_name: str = llm_name
        self.llm_temperature = llm_temperature
        self.system_prompt: str = system_prompt

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        messages: list[dict] = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ]
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=messages,
        )
        messages.append(
            {
                "role": llm_api_response.choices[0].message.role,
                "content": llm_api_response.choices[0].message.content,
            }
        )
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[ChatMessage(**msg) for msg in messages[1:]],
                all_messages=[ChatMessage(**msg) for msg in messages],
                token_usage=llm_api_response.usage.model_dump(),
            )
        )

    def view_memory_as_json(self) -> dict:
        """
        Render current state of the agent's memory as a dict
        """
        return {"note": "This LLM has no memory (each chat completion is new)"}
