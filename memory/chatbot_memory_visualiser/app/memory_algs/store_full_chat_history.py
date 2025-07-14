"""
Memory algorithm which simply stores the entire chat history
"""

import json

import openai
from loguru import logger

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


class StoreFullChatHistory(MemoryAlg):
    """
    Memory algorithm which just stores the entire chat history
    """

    def __init__(
        self,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
        system_prompt: str,
    ) -> None:
        self.chat_history: list[ChatMessageDetail] = []
        self.llm_client = llm_client
        self.llm_name: str = llm_name
        self.llm_temperature = llm_temperature
        self.system_prompt: str = system_prompt

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

        logger.debug(
            "\n" + json.dumps([x.model_dump() for x in self.chat_history], indent=4)
        )

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        chat_msgs_to_llm: list[dict] = (
            [msg.model_dump() for msg in self.chat_history[-1].all_messages]
            if self.chat_history
            else []
        )
        user_msg = ChatMessage(
            role="user",
            content=user_msg,
        )
        chat_msgs_to_llm.append(user_msg.model_dump())
        llm_api_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=chat_msgs_to_llm,
        )
        llm_reply = ChatMessage(
            role=llm_api_response.choices[0].message.role,
            content=llm_api_response.choices[0].message.content,
        )
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[user_msg, llm_reply],
                all_messages=[ChatMessage(**msg) for msg in chat_msgs_to_llm]
                + [llm_reply],
                token_usage=llm_api_response.usage.model_dump(),
            )
        )
        logger.debug(
            "\n" + json.dumps([x.model_dump() for x in self.chat_history], indent=4)
        )

    def view_memory_as_json(self) -> dict:
        """
        Render latest state of the agent's memory as a dict
        """
        return self.chat_history[-1].model_dump()
