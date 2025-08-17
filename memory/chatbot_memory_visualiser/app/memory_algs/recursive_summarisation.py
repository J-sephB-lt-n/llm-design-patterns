"""
Memory algorithm which maintains a concise summary of the full chat history
"""

import json
from textwrap import dedent
from typing import Final, Literal

import openai
from loguru import logger

from app.interfaces.memory_alg_protocol import ChatMessage, ChatMessageDetail, MemoryAlg


MEMORY_ITERATION_PROMPT: Final[str] = dedent(
    """
    <previous-chat-summary>
    {previous_chat_summary}
    </previous-chat-summary>

    <most-recent-interactions>
    {most_recent_interactions}
    </most-recent-interactions>

    Your task is to update the previous chat summary (a concise summary of your entire past \
    conversation with the user) with the new facts and information present in your most \
    recent interactions with the user.
    
    Do your best to pack as much information into the summary as you can.
    If (due to space constraints) you must discard information, then discard information which \
    you deem to be the least important to the user and the conversation.

    Your summary alone will be used by a future assistant (with no context other than your \
    summary) in order to continue the conversation with the user, so your summary must \
    contain all of the information essential to keeping track of the important parts of the \
    conversation.

    Your summary is not allowed to be longer than {summary_max_n_sentences} sentences.

    Return only the summary text - no commentary. You may return an empty string.
"""
)


class RecursiveSummarisation(MemoryAlg):
    """
    Memory algorithm which keeps the few most recent chat messages and a concise summary of the \
full chat history.

    Based very loosely on the algorithm described in the paper "Recursively Summarizing Enables \
Long-Term Dialogue Memory in Large Language Models" (https://arxiv.org/abs/2308.15022)

    Attributes:
        llm_client (openai.OpenAI): Model API client
        llm_name (str): Model name (model identifier in model API)
        llm_temperature (float): Model temperature (level of model output randomness)
        summary_max_n_sentences (int): Model is instructed to keep the length of the full chat \
                                        history summary to at most this many sentences.
        summarise_every_n_user_messages (int): Model updates it's summary of the full chat \
                                        history at this frequency.
                                        At this point, messages are removed from "session memory" \
                                        (in prompt) and their contents used to update the summary.
        min_n_messages_in_session_memory (int): When messages are removed from "session memory" \
                                        (in prompt) and included in the summary, this many \
                                        messages (the most recent) are kept in session memory.
        message_render_style (str): Controls how chat messages are rendered when including them in \
            model prompts.
            One of ['json_dumps', 'plain_text', 'xml'].
            'plain_text' looks like "USER: ...<br>ASSISTANT: ...<br> ..."
            'json_dumps' gives standard chat completion messages JSON
            'xml' looks like "<user>...</user> <assistant>...</assistant> ..."
    """

    alg_description = dedent(
        """
        Periodically updates a short summary of the full chat history in order to reduce the context size.
        """
    )

    def __init__(
        self,
        llm_client: openai.OpenAI,
        llm_name: str,
        llm_temperature: float,
        system_prompt: str = "You are a helpful and creative assistant conversing with a user.",
        summary_max_n_sentences: int = 20,
        summarise_every_n_user_messages: int = 10,
        min_n_messages_in_session_memory: int = 5,
        messge_render_style: Literal["json_dumps", "plain_text", "xml"] = "plain_text",
    ):
        self.chat_history: list[ChatMessageDetail] = []
        self.llm_client = llm_client
        self.llm_name = llm_name
        self.llm_temperature = llm_temperature
        self.system_prompt = system_prompt
        self.summary_max_n_sentences = summary_max_n_sentences
        self.summarise_every_n_user_messages = summarise_every_n_user_messages
        self.min_n_messages_in_session_memory = min_n_messages_in_session_memory
        self.message_render_style = messge_render_style
        self.session_memory: list[ChatMessage] = []
        self.chat_summary: str = ""
        self.user_message_counter: int = 0

    def chat_messages_to_text(
        self,
        messages: list[ChatMessage],
        message_render_style: Literal["json_dumps", "plain_text", "xml"],
    ) -> str:
        """
        Represent sequence of chat-completion messages as a single string
        """
        match message_render_style:
            case "json_dumps":
                return json.dumps(
                    [msg.model_dump() for msg in messages],
                    indent=4,
                )
            case "plain_text":
                return "\n".join(
                    [f"{msg.role.upper()}: {msg.content}" for msg in messages]
                )
            case "xml":
                return "\n".join(
                    [f"<{msg.role}>\n{msg.content}\n</{msg.role}>" for msg in messages]
                )

    def chat(self, user_msg: str) -> None:
        """
        Process a new user message, update memory and chat history
        """
        user_message = ChatMessage(
            role="user",
            content=user_msg,
        )
        self.session_memory.append(user_message)
        internal_generation_prompt_messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=self.system_prompt
                + f"""
                
<summary-of-chat-history>
{self.chat_summary}
</summary-of-chat-history>
                """,
            ),
            *self.session_memory,
        ]
        api_generation_response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            temperature=self.llm_temperature,
            messages=[msg.model_dump() for msg in internal_generation_prompt_messages],
        )
        assistant_generation_response = ChatMessage(
            role=api_generation_response.choices[0].message.role,
            content=api_generation_response.choices[0].message.content,
        )
        logger.debug(
            "\n".join(
                f"--{msg.role.upper()}--\n{msg.content}"
                for msg in [
                    *internal_generation_prompt_messages,
                    assistant_generation_response,
                ]
            )
        )
        self.session_memory.append(assistant_generation_response)
        self.chat_history.append(
            ChatMessageDetail(
                visible_messages=[user_message, assistant_generation_response],
                all_messages=[
                    *internal_generation_prompt_messages,
                    assistant_generation_response,
                ],
                token_usage=api_generation_response.usage.model_dump(),
            )
        )
        self.user_message_counter += 1
        if self.user_message_counter >= self.summarise_every_n_user_messages:
            self.user_message_counter = 0
            internal_summarisation_prompt_messages: list[ChatMessage] = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(
                    role="user",
                    content=MEMORY_ITERATION_PROMPT.format(
                        summary_max_n_sentences=self.summary_max_n_sentences,
                        previous_chat_summary=self.chat_summary,
                        most_recent_interactions=self.chat_messages_to_text(
                            (
                                self.session_memory
                                if self.min_n_messages_in_session_memory == 0
                                else self.session_memory[
                                    : -self.min_n_messages_in_session_memory
                                ]
                            ),
                            message_render_style=self.message_render_style,
                        ),
                    ),
                ),
            ]
            api_summarisation_response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                temperature=self.llm_temperature,
                messages=[
                    msg.model_dump() for msg in internal_summarisation_prompt_messages
                ],
            )
            assistant_summarisation_response = ChatMessage(
                role=api_summarisation_response.choices[0].message.role,
                content=api_summarisation_response.choices[0].message.content,
            )
            logger.debug(
                "\n".join(
                    f"--{msg.role.upper()}--\n{msg.content}"
                    for msg in [
                        *internal_summarisation_prompt_messages,
                        assistant_summarisation_response,
                    ]
                )
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
                        *internal_summarisation_prompt_messages,
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
            "short_term_chat_history": [
                msg.model_dump() for msg in self.session_memory
            ],
        }
