import os
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessage

from app.prompts.loader import render_system_prompt

class LLMClient:
    def __init__(self,tools:list[ChatCompletionFunctionToolParam]=None):
        load_dotenv()
        self._main_model = 'gpt-4o-mini'
        self._main_client = self._build_client('https://api.openai.com/v1',os.getenv('OPENAI_API_KEY'))
        self._messages_history = self._init_messages_history()
        self._tools = tools

    @staticmethod
    def _build_client(base_url: str, api_key: str) -> OpenAI:
        return OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _init_messages_history() -> list[dict[str,str]]:
        return [{
            'role':'system',
            'content':render_system_prompt()
        }]

    def _build_messages(self, message: dict[str,str]) -> list[dict[str,str]]:
        self._messages_history.append(message)
        return self._messages_history

    def send(self, message: dict[str,str])-> tuple[ChatCompletionMessage, int]:
        messages = self._build_messages(message)
        response = self._main_client.chat.completions.create(
            model=self._main_model,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
            tools=self._tools,
            tool_choice="auto"
        )
        answer = response.choices[0].message
        self._build_messages(answer)
        return answer, response.usage.total_tokens