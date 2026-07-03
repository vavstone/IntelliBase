import tiktoken
from typing import List, Dict

def count_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> int:
    """
    Точный подсчёт токенов для сообщений в формате OpenAI Chat API.
    Использует tiktoken и учитывает служебные токены.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback для неизвестных моделей
        encoding = tiktoken.get_encoding("cl100k_base")

    # Специальные токены для ChatML
    tokens_per_message = 3  # <|im_start|>, role, <|im_end|>
    tokens_per_name = 1     # дополнительный токен для поля "name"

    num_tokens = 0
    for msg in messages:
        num_tokens += tokens_per_message
        for key, value in msg.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # завершающий токен <|im_start|>assistant
    return num_tokens