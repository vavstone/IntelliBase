import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor


def setup_tracing(project_name:str = "diploma-fastapi") -> None:
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT","http://phoenix:4317")
    tracer_provider = register(project_name=project_name, endpoint=endpoint)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)