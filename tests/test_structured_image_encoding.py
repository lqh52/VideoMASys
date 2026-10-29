"""Real image encoding with synthetic pixels; no network, weights or GPU calls."""

import base64
import hashlib
import io
import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from videomasys.zerogpu import inference
from videomasys.zerogpu.openai import OpenAI41MiniVLM
from videomasys.zerogpu.remote import HostedVLM, HostedVLMError
from videomasys.zerogpu.remote_runtime import GPT41_MINI_MODEL, RemotePipeline


@pytest.fixture
def pictures(tmp_path):
    paths = []
    for index in range(6):
        path = tmp_path / f"fixture_{index}.png"
        with Image.new("RGB", (1280, 720), (index * 30, 60, 170)) as image:
            image.save(path)
        paths.append(str(path))
    return paths


def encoded_images(adapter, paths):
    payload, _ = adapter._payload("Describe only these fixture images.", paths, 80)
    content = json.loads(payload)["messages"][1]["content"]
    images = []
    for item in content:
        if item["type"] == "image_url":
            data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(data)) as image:
                images.append((image.size, hashlib.sha256(data).hexdigest()))
    return images


@pytest.mark.parametrize("adapter_type", [HostedVLM, OpenAI41MiniVLM])
def test_fixed_bounds_keep_actual_wire_jpegs_identical_in_six_and_eighteen_frame_packs(
    adapter_type, pictures,
):
    transport = httpx.MockTransport(lambda request: pytest.fail("No HTTP request allowed"))
    with httpx.Client(transport=transport) as client:
        adapter = adapter_type("fixture-noncredential", client=client, image_bounds=(512, 384))
        try:
            six = encoded_images(adapter, pictures)
            eighteen = encoded_images(adapter, pictures * 3)
        finally:
            adapter.close()
    assert len(six) == 6 and len(eighteen) == 18
    assert eighteen == six * 3
    assert all(size == (512, 288) for size, _ in eighteen)


@pytest.mark.parametrize("bounds", [
    [512, 384], (512,), (True, 384), (0, 384), (641, 384), (512, 481),
])
def test_invalid_bounds_fail_before_client_construction(bounds, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: pytest.fail("Client constructed"))
    with pytest.raises(HostedVLMError) as failure:
        HostedVLM("fixture-noncredential", image_bounds=bounds)
    assert failure.value.code == "invalid_request"


def test_openai_runtime_passes_structured_image_bounds(monkeypatch):
    # API-provider/default-profile forwarding is covered in test_zerogpu_remote_runtime.
    monkeypatch.setenv("VLM_MODEL", GPT41_MINI_MODEL)
    allowance = SimpleNamespace(before_request=Mock(), close=Mock())
    guard = SimpleNamespace(start_run=Mock(return_value=allowance), snapshot=lambda: {})
    received = []

    def factory(token, *, model, before_request, **kwargs):
        received.append(kwargs)
        return SimpleNamespace(calls=[], close=Mock())

    pipeline = RemotePipeline(
        provider="openai", token="fixture-noncredential", guard=guard,
        embed=Mock(), vlm_factory=factory, prompt_profile="structured",
    )
    with pipeline.analysis_session({}) as active:
        assert active.prompt_profile == "structured"
    assert received == [{"image_bounds": (512, 384)}]
    allowance.before_request.assert_not_called()
    allowance.close.assert_called_once()


def mock_gpu_imports(monkeypatch, captured):
    """Exercise the real local generate closure without importing GPU libraries."""
    class Inputs(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            pictures = [item["image"] for item in messages[1]["content"]
                        if item["type"] == "image"]
            captured.append([(picture.size, hashlib.sha256(picture.tobytes()).hexdigest())
                             for picture in pictures])
            return Inputs(input_ids=SimpleNamespace(shape=(1, 5)))

        @staticmethod
        def batch_decode(outputs, **kwargs):
            return ["fixture response"]

    class Outputs:
        def __getitem__(self, key):
            return []

    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="fixture-revision"),
                            generate=lambda **kwargs: Outputs())
    model.eval = lambda: model
    modules = {name: ModuleType(name) for name in (
        "spaces", "torch", "sentence_transformers", "transformers",
    )}
    modules["spaces"].GPU = lambda **kwargs: lambda function: function
    modules["torch"].set_num_threads = lambda value: None
    modules["torch"].bfloat16 = "fixture-bfloat16"
    modules["torch"].inference_mode = nullcontext
    modules["sentence_transformers"].SentenceTransformer = lambda *a, **kw: SimpleNamespace(
        encode=lambda *a, **kw: [],
    )
    modules["transformers"].AutoProcessor = SimpleNamespace(
        from_pretrained=lambda *a, **kw: Processor(),
    )
    modules["transformers"].Qwen3VLForConditionalGeneration = SimpleNamespace(
        from_pretrained=lambda *a, **kw: model,
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("profile", ["clean", "structured"])
def test_local_generate_captures_structured_profile_and_reuses_identical_pixels(
    profile, pictures, monkeypatch,
):
    captured = []
    mock_gpu_imports(monkeypatch, captured)
    monkeypatch.setenv("VLM_PROMPT_PROFILE", profile)
    pipeline, _, _ = inference.load_runtime()
    # Later environment changes cannot alter this loaded runtime's image profile.
    monkeypatch.setenv("VLM_PROMPT_PROFILE", "clean" if profile == "structured" else "structured")
    assert pipeline.generate("Fixture prompt", pictures, 80) == "fixture response"
    assert pipeline.generate("Fixture prompt", pictures * 3, 80) == "fixture response"
    six, eighteen = captured
    if profile == "structured":
        assert pipeline.prompt_profile == "structured"
        assert eighteen == six * 3
        assert all(size == (512, 288) for size, _ in six)
    else:
        assert pipeline.prompt_profile == "clean"  # Preserve this loader's earlier default.
        assert all(size == (640, 360) for size, _ in six)
        assert all(size == (512, 288) for size, _ in eighteen)
