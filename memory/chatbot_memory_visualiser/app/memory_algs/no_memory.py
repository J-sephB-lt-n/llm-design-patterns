"""
Memory algorithm with no memory (baseline)
"""

import openai

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


class NoMemory(MemoryAlg):
    """
    Memory algorithm with no memory (baseline)
    """

    def __init__(
        self,
        system_prompt: str | None,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
    ) -> None:
        self.chat_history: list[ChatMessageDetail] = []
        self.llm_client = llm_client
        self.llm_name: str = llm_name
        self.llm_temperature = llm_temperature
        self.system_prompt: str | None = system_prompt

        if self.system_prompt:
            self.chat_history.append(
                ChatMessageDetail(
                    visible_messages=[],
                    all_messages=[
                        ChatMessage(
                            role="system",
                            content=system_prompt,
                        )
                    ],
                    token_usage={},
                )
            )

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        ...

    def view_memory_as_json(self) -> dict:
        """
        Render current state of the agent's memory as a dict
        """
        return {"note": "This LLM has no memory (each chat completion is new)"}
