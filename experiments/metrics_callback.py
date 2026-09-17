from langchain_core.callbacks import BaseCallbackHandler

class MetricsCallback(BaseCallbackHandler):
    """Считает LLM-вызовы и токены по всем агентам графа."""
    def __init__(self) -> None:
        self.llm_calls = 0
        self.total_tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.llm_calls += 1

    def on_llm_end(self, response, **kwargs) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    self.total_tokens += usage.get("total_tokens", 0)


