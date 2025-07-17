"""
Memory algorithm which maintains a concise summary of the full chat history
"""

import json
from textwrap import dedent
from typing import Final, Literal

import openai

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
    exceed {summary_max_n_sentences} sentences.
    Remember, the memory should serve as a reference point to maintain continuity in the dialogue \
    and help you respond accurately to the user based on their personality.

    <previous-memory>
    {previous_memory}
    </previous-memory>

    <session-context>
    {session_context}
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
    {previous_memory}
    </previous-memory>

    <current-context>
    {current_context}
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
        summary_max_n_sentences: int = 20,
        summarise_every_n_user_messages: int = 10,
        min_n_messages_in_session_memory: int = 5,
        session_memory_render_style: Literal["json_dumps", "plain_text"] = "plain_text",
    ):
        self.chat_history: list[ChatMessageDetail] = []
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.summary_max_n_sentences = summary_max_n_sentences
        self.summarise_every_n_user_messages = summarise_every_n_user_messages
        self.min_n_messages_in_session_memory = min_n_messages_in_session_memory
        self.session_memory_render_style = session_memory_render_style
        self.session_memory: list[ChatMessage] = []
        self.chat_summary: str = ""
        self.user_message_counter: int = 0

    def chat_messages_to_text(
        self,
        messages: list[ChatMessage],
        output_style: Literal["json_dumps", "plain_text"],
    ) -> str:
        """
        Represent sequence of chat-completion messages as a single string
        """
        match output_style:
            case "json_dumps":
                return json.dumps(
                    [msg.model_dump() for msg in messages],
                    indent=4,
                )
            case "plain_text":
                return "\n".join(
                    [f"{msg.role.upper()}: {msg.content}" for msg in messages]
                )
            case _:
                raise ValueError(f"Unknown output style '{output_style}'")

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        user_message = ChatMessage(
            role="user",
            content=user_msg,
        )
        self.session_memory.append(user_message)
        internal_generation_prompt = ChatMessage(
            role="user",
            content=MEMORY_BASED_RESPONSE_GENERATION_PROMPT.format(
                previous_memory=self.chat_summary,
                current_context=self.chat_messages_to_text(
                    self.session_memory,
                    output_style=self.session_memory_render_style,
                ),
            ),
        )
        api_generation_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[internal_generation_prompt.model_dump()],
        )
        assistant_generation_response = ChatMessage(
            role=api_generation_response.choices[0].message.role,
            content=api_generation_response.choices[0].message.content,
        )
        self.session_memory.append(assistant_generation_response)
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[user_message, assistant_generation_response],
                all_messages=[
                    internal_generation_prompt,
                    assistant_generation_response,
                ],
                token_usage=api_generation_response.usage.model_dump(),
            )
        )
        self.user_message_counter += 1
        if self.user_message_counter >= self.summarise_every_n_user_messages:
            self.user_message_counter = 0
            internal_summarisation_prompt = ChatMessage(
                role="user",
                content=MEMORY_ITERATION_PROMPT.format(
                    summary_max_n_sentences=self.summary_max_n_sentences,
                    previous_memory=self.chat_summary,
                    session_context=self.chat_messages_to_text(
                        (
                            self.session_memory
                            if self.min_n_messages_in_session_memory == 0
                            else self.session_memory[
                                : -self.min_n_messages_in_session_memory
                            ]
                        ),
                        output_style=self.session_memory_render_style,
                    ),
                ),
            )
            api_summarisation_response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                temperature=self.llm_temperature,
                messages=[internal_summarisation_prompt.model_dump()],
            )
            assistant_summarisation_response = ChatMessage(
                role=api_summarisation_response.choices[0].message.role,
                content=api_summarisation_response.choices[0].message.content,
            )
            self.session_memory = (
                []
                if self.min_n_messages_in_session_memory == 0
                else self.session_memory[-self.min_n_messages_in_session_memory :]
            )
            self.chat_summary = assistant_summarisation_response.content
            self.chat_history.append(
                ChatMessageDetail(
                    visible_messages=[],
                    all_messages=[
                        internal_summarisation_prompt,
                        assistant_summarisation_response,
                    ],
                    token_usage=api_summarisation_response.usage.model_dump(),
                )
            )

    def view_memory_as_json(self) -> dict:
        """
        Render latest state of the agent's memory as a dict
        """
        return {
            "long_term_summary": self.chat_summary,
            "short_term_chat_history": self.session_memory,
        }
