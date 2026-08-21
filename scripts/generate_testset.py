"""Генерация golden dataset через RAGAS TestsetGenerator (группа eval).

RAGAS строит граф знаний из документов корпуса и генерирует разнотипные вопросы
(single-hop / multi-hop / abstract) с эталонным ответом и эталонными контекстами.
LLM генератора — DeepSeek (OpenAI-совместимый, без VPN), эмбеддинги — локальная E5.
Сырой результат сохраняется в CSV — дальше обязательна ручная вычитка (выкинуть
дубли и слишком общие вопросы, доразметить reference).

Запуск:
    uv run --extra eval python scripts/generate_testset.py --size 40
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import Document, SimpleDirectoryReader  # noqa: E402
from llama_index.core.base.llms.types import LLMMetadata, MessageRole  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.llms.openai import OpenAI as _OpenAI  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402
from ragas.testset.synthesizers.single_hop.specific import (  # noqa: E402
    SingleHopSpecificQuerySynthesizer,
)

from app.core.config import get_settings  # noqa: E402
from app.services.chunking import build_embed_model  # noqa: E402

# Русскоязычная подсказка для генератора: корпус домена ФТС — на русском.
RU_CONTEXT = "Генерируй вопросы и ответы строго на русском языке."


class DeepSeekLLM(_OpenAI):
    """OpenAI-совместимый LLM на DeepSeek (без VPN).

    Базовый llama_index.llms.openai.OpenAI определяет context_window по имени
    модели через openai_modelname_to_contextsize и падает на имени `deepseek-*`
    (не из списка OpenAI). Переопределяем metadata явно — тот же приём, что
    OllamaLLM в app/services/rag.py.
    """

    def __init__(self, *args, context_window: int = 16384, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._context_window = context_window

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


def _build_testset_llm(settings) -> DeepSeekLLM:
    """DeepSeek-LLM для генератора вопросов (OpenAI-совместимый эндпоинт).

    `additional_kwargs` прокидывает extra_body в каждый вызов: отключаем
    chain-of-thought (reasoning_tokens) — для extract-задач это оверхед.
    """
    return DeepSeekLLM(
        model=settings.eval_judge_model,
        api_base=settings.llm.deepseek_base_url,
        api_key=settings.llm.deepseek_api_key.get_secret_value(),
        temperature=0.0,
        additional_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS TestsetGenerator")
    parser.add_argument("--size", type=int, default=40, help="число пар Q/A")
    parser.add_argument(
        "--out",
        default="tests/eval/golden_dataset_raw.csv",
        help="куда писать CSV",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=600,
        help="потолок чанков для построения графа знаний (равномерная выборка)",
    )
    args = parser.parse_args()

    settings = get_settings()
    docs = SimpleDirectoryReader(
        str(settings.rag_data_dir), recursive=True
    ).load_data()
    print(f"Loaded {len(docs)} documents")

    # Предварительная нарезка на ~300-токенные чанки. Причина: default_transforms
    # RAGAS на смешанном корпусе (крупные многостраничные PDF + короткие страницы)
    # выбирает ветку с HeadlineSplitter, который падает на страницах без 'headlines'
    # (баг ragas 0.4.3). Равномерные чанки 101–500 токенов уводят в ветку без
    # сплиттера. Заодно это ближе к тому, как сам RAG индексирует корпус (512/64).
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=40)
    chunked = [
        Document(text=node.get_content(), metadata=node.metadata or {})
        for node in splitter.get_nodes_from_documents(docs)
    ]
    print(f"Chunked to {len(chunked)} chunks (~300 tokens)")

    # Построение графа знаний — O(чанков) LLM-вызовов; на полном корпусе это часы.
    # Ограничиваем равномерной выборкой, чтобы покрыть корпус, а не один алфавитный
    # префикс (иначе попали бы только в malahit/raznoe).
    if args.max_chunks and len(chunked) > args.max_chunks:
        step = max(1, len(chunked) // args.max_chunks)
        chunked = chunked[::step][: args.max_chunks]
        print(f"Sampled {len(chunked)} chunks evenly (max_chunks={args.max_chunks})")

    # from_llama_index оборачивает LLM и эмбеддинги LlamaIndex под генератор.
    # Эмбеддинги — локальная E5 с префиксами passage:/query: (build_embed_model).
    generator = TestsetGenerator.from_llama_index(
        llm=_build_testset_llm(settings),
        embedding_model=build_embed_model(settings.embedding.model),
    )

    # Только single-hop: multi-hop/abstract на flash-модели генерируют вырожденные
    # зацикленные вопросы (упёрлись в max_tokens и потеряли reference/contexts).
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator.llm, llm_context=RU_CONTEXT), 1.0),
    ]
    testset = generator.generate_with_llamaindex_docs(
        chunked,
        testset_size=args.size,
        query_distribution=query_distribution,
    )

    df = testset.to_pandas()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    # Эталон: user_input / reference / reference_contexts. Поля retrieved_contexts
    # и response добавит run_eval.py при прогоне своего RAG.
    print(df[["user_input", "reference", "reference_contexts"]].head())
    print(f"\nСохранено: {out} ({len(df)} строк). Дальше — ручная вычитка.")


if __name__ == "__main__":
    main()
