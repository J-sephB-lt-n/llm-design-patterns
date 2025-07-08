"""
Memory algorithm which simply stores the entire chat history
"""

import openai

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


class StoreFullChatHistory(MemoryAlg):
    """
    Memory algorithm which just stores the entire chat history
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
        chat_msgs_to_llm: list[dict] = [
            msg.model_dump() for msg in self.chat_history[-1].all_messages
        ]
        user_msg = ChatMessage(
            role="user",
            content=user_msg,
        )
        chat_msgs_to_llm.append(user_msg.model_dump())
        llm_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=chat_msgs_to_llm,
        )
        chat_msgs_to_llm.append(
            {
                "role": llm_response.choices[0].message.role,
                "content": llm_response.choices[0].message.content,
            }
        )
