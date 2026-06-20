import pytest
from pydantic import ValidationError
from app.schemas.chat import ChatRequest, ChatResponse


def test_chat_request_valid():
    req = ChatRequest(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}
        ],
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=100
    )
    assert req.model == "gpt-4o-mini"
    assert len(req.messages) == 2


def test_chat_request_first_message_not_assistant():
    with pytest.raises(ValidationError, match="Первое сообщение не может быть от assistant"):
        ChatRequest(
            messages=[{"role": "assistant", "content": "I'm assistant"}],
            model="gpt-4o-mini"
        )


def test_chat_request_message_min_length():
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        ChatRequest(
            messages=[{"role": "user", "content": ""}],
            model="gpt-4o-mini"
        )


def test_chat_request_message_max_length():
    long_content = "x" * 100_001
    with pytest.raises(ValidationError, match="String should have at most 100000 characters"):
        ChatRequest(
            messages=[{"role": "user", "content": long_content}],
            model="gpt-4o-mini"
        )


def test_chat_response_from_openai(mocker):
    # Мокаем объект ответа OpenAI
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = "Hello"
    mock_choice.finish_reason = "stop"

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30

    mock_raw = mocker.MagicMock()
    mock_raw.choices = [mock_choice]
    mock_raw.model = "gpt-4o-mini"
    mock_raw.usage = mock_usage

    response = ChatResponse.from_openai(mock_raw)
    assert response.content == "Hello"
    assert response.model == "gpt-4o-mini"
    assert response.finish_reason == "stop"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 30