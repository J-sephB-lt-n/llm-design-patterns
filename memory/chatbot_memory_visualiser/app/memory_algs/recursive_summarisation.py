"""
Memory algorithm which maintains a concise summary of the full chat history
"""

import json
from textwrap import dedent
from typing import Final

import openai
from loguru import logger

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


MEMORY_ITERATION_PROMPT: Final[str] = dedent(
    # This is an edited version of the prompt in the paper
    #   "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models"
    #   (https://arxiv.org/abs/2308.15022)
    """
    You are an advanced AI language model with the ability to store and update a memory to keep \
    track of key personality information for both the user and the bot. \
    You will receive a previous memory and dialogue context. \
    Your goal is to update the memory by incorporating the new personality information.
    To successfully update the memory, follow these steps:
    1. Carefully analyze the existing memory and extract the key personality of the user and bot from it.
    2. Consider the dialogue context provided to identify any new or changed personality that needs to \
        be incorporated into the memory.
    3. Combine the old and new personality information to create an updated representation of the user and \
        bot's traits.
    4. Structure the updated memory in a clear and concise manner, ensuring it does not \
    exceed {{ n_sentences }} sentences.
    Remember, the memory should serve as a reference point to maintain continuity in the dialogue \
    and help you respond accurately to the user based on their personality.

    <previous-memory>
    {{ Previous Memory }}
    </previous-memory>

    <session-context>
    {{ Session Context }}
    </session-context>

    Return only the updated memory text.
"""
)

MEMORY_BASED_RESPONSE_GENERATION_PROMPT: Final[str] = dedent(
    # This is an edited version of the prompt in the paper
    #   "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models"
    #   (https://arxiv.org/abs/2308.15022)
    """
    You will be provided with a memory containing personality information for both yourself \
    and the user.
    Your goal is to respond accurately to the user based on the personality traits and dialogue context.
    Follow these steps to successfully complete the task:
    1. Analyze the provided memory to extract the key personality traits for both yourself and the user. 
    2. Review the dialogue history to understand the context and flow of the conversation. 
    3. Utilize the extracted personality traits and dialogue context to formulate an appropriate response. 
    4. If no specific personality trait is applicable, respond naturally as a human would. 
    5. Pay attention to the relevance and importance of the personality information, focusing on capturing
    the most significant aspects while maintaining the overall coherence of the memory.

    <previous-memory>
    {{ previous_memory }}
    </previous-memory>

    <current-context>
    {{ current_context }}
    </current-context>
    """
)


class RecursiveSummarisation(MemoryAlg):
    """
    Memory algorithm which keeps only a concise summary of the full chat history
    """

    alg_description = dedent(
        """
        Periodically updates a short summary of the full chat history in order to reduce the context size.
        Based very loosely on the algorithm described in the paper 
        "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models"
        (https://arxiv.org/abs/2308.15022)
        """
    )

    def __init__(
        self,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
        summary_len_n_sentences: int,
        summarise_every_n_user_messages: int,
    ):
        self.chat_history: list[ChatMessageDetail] = []
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.summary_len_n_sentences = summary_len_n_sentences
        self.summarise_every_n_user_messages = summarise_every_n_user_messages

    def chat(self, user_msg: str) -> None:
        """
        TODO
        """
        logger.debug(
            "\n" + json.dumps([x.model_dump() for x in self.chat_history], indent=4)
        )
