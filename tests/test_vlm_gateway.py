from videomasys.model_gateway.vlm import VLMGateway


def test_strip_json_markdown_fence() -> None:
    value = "```json\n{\"answer\": \"ok\"}\n```"
    assert VLMGateway._strip_fences(value) == '{"answer": "ok"}'


def test_plain_json_is_unchanged() -> None:
    value = '{"answer": "ok"}'
    assert VLMGateway._strip_fences(value) == value
