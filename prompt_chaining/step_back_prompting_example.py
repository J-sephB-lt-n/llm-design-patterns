"""
An example of a 2-step prompt chain.

- The LLM is first asked a general question, designed to elicit some of it's base knowledge \
    (this is known as "step-back" prompting)
- The LLM is then asked to do a specific task (guided by the previously generated information)

In this example, the LLM is first asked to list the core elements of software best \
practice, and then to review the code in this python script.
"""

import os

import dotenv
import openai

dotenv.load_dotenv(".env")

llm_client = openai.OpenAI(
    base_url=os.environ["OPENAI_API_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

if __name__ == "__main__":
    chat_history = [
        {
            "role": "system",
            "content": "You are an experienced software architect",
        },
        {
            "role": "user",
            "content": "What are the core tenets of software best practice? List them.",
        },
    ]

    llm_software_tenets_response = llm_client.chat.completions.create(
        model="azure.gpt-4o",
        temperature=0,
        messages=chat_history,
        max_completion_tokens=2_000,
    )

    chat_history.append(
        {
            "role": llm_software_tenets_response.choices[0].message.role,
            "content": llm_software_tenets_response.choices[0].message.content,
        }
    )

    with open("./prompt_chaining/step_back_prompting_example.py", "r") as file:
        code_contents: str = file.read()

    chat_history.append(
        {
            "role": "user",
            "content": f"""
Please review the following python script in terms of the core tenets:
```python
{code_contents}
```
            """.strip(),
        }
    )

    software_review_response = llm_client.chat.completions.create(
        model="azure.gpt-4o",
        temperature=0,
        messages=chat_history,
        max_completion_tokens=5_000,
    )

    chat_history.append(
        {
            "role": software_review_response.choices[0].message.role,
            "content": software_review_response.choices[0].message.content,
        }
    )

    for message in chat_history:
        print("---", message["role"].upper(), "---")
        print(message["content"])
        print()
