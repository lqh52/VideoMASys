"""Entrypoint tests use isolated runtime stubs: no models, API calls, or Gradio server."""

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy/zerogpu/app.py"


@pytest.fixture
def startup(monkeypatch, tmp_path):
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    monkeypatch.delenv("VIDEOMASYS_UI_ONLY", raising=False)
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    ui = ModuleType("videomasys.zerogpu.ui")
    ui.STYLE, ui.THEME = "fixture-style", "fixture-theme"
    demo = SimpleNamespace(launch=Mock())
    ui.build_demo = Mock(return_value=demo)
    api = ModuleType("videomasys.zerogpu.remote_runtime")
    api.load_runtime = Mock(return_value=("api-pipeline", "api-index", "api-qa"))
    local = ModuleType("videomasys.zerogpu.inference")
    local.load_runtime = Mock(return_value=("local-pipeline", "local-index", "local-qa"))
    for module in (ui, api, local, ModuleType("spaces")):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    imported = []
    real_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        imported.append(name)
        if name.split(".")[0] in {
            "torch",
            "torchvision",
            "transformers",
            "sentence_transformers",
            "accelerate",
        }:
            raise AssertionError("Entrypoint must not import heavyweight model dependencies")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    return SimpleNamespace(ui=ui, api=api, local=local, demo=demo, imported=imported)


@pytest.mark.parametrize("backend", [None, "api", " API "])
def test_default_cpu_entrypoint_uses_api_without_spaces(startup, monkeypatch, backend):
    if backend is not None:
        monkeypatch.setenv("VLM_BACKEND", backend)
    runpy.run_path(str(APP), run_name="__main__")
    startup.api.load_runtime.assert_called_once_with()
    startup.local.load_runtime.assert_not_called()
    startup.ui.build_demo.assert_called_once_with("api-pipeline", "api-index", "api-qa")
    assert "spaces" not in startup.imported
    assert "videomasys.zerogpu.inference" not in startup.imported
    startup.demo.launch.assert_called_once()
    assert startup.demo.launch.call_args.kwargs["allowed_paths"] == ["space_library"]


def test_local_backend_is_explicit_and_imports_spaces_before_runtime(startup, monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "local")
    runpy.run_path(str(APP), run_name="__main__")
    startup.local.load_runtime.assert_called_once_with()
    startup.api.load_runtime.assert_not_called()
    startup.ui.build_demo.assert_called_once_with("local-pipeline", "local-index", "local-qa")
    assert startup.imported.index("spaces") < startup.imported.index("videomasys.zerogpu.ui")
    assert "videomasys.zerogpu.remote_runtime" not in startup.imported


@pytest.mark.parametrize("backend", ["openai", " OPENAI "])
def test_openai_backend_is_explicit_and_cpu_only(startup, monkeypatch, backend):
    monkeypatch.setenv("VLM_BACKEND", backend)
    runpy.run_path(str(APP), run_name="__main__")
    startup.api.load_runtime.assert_called_once_with(provider="openai")
    startup.local.load_runtime.assert_not_called()
    startup.ui.build_demo.assert_called_once_with("api-pipeline", "api-index", "api-qa")
    assert "spaces" not in startup.imported
    assert "videomasys.zerogpu.inference" not in startup.imported


@pytest.mark.parametrize("backend", [None, "local", "invalid"])
def test_ui_preview_never_initializes_either_runtime(startup, monkeypatch, backend):
    monkeypatch.setenv("VIDEOMASYS_UI_ONLY", "1")
    if backend is not None:
        monkeypatch.setenv("VLM_BACKEND", backend)
    runpy.run_path(str(APP), run_name="__main__")
    startup.ui.build_demo.assert_called_once_with()
    startup.api.load_runtime.assert_not_called()
    startup.local.load_runtime.assert_not_called()
    assert "spaces" not in startup.imported


@pytest.mark.parametrize("backend", ["", "automatic", "gpu", "https://example.invalid"])
def test_unknown_backend_fails_without_loading_or_falling_back(startup, monkeypatch, backend):
    monkeypatch.setenv("VLM_BACKEND", backend)
    with pytest.raises(ValueError, match="VLM_BACKEND"):
        runpy.run_path(str(APP), run_name="__main__")
    startup.ui.build_demo.assert_not_called()
    startup.api.load_runtime.assert_not_called()
    startup.local.load_runtime.assert_not_called()
    assert "spaces" not in startup.imported


def test_cpu_requirements_are_lightweight_and_python312_compatible():
    requirements = [
        Requirement(line)
        for line in (ROOT / "deploy/zerogpu/requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    indexed = {item.name.lower().replace("_", "-"): item for item in requirements}
    assert not (
        {
            "torch",
            "torchvision",
            "transformers",
            "sentence-transformers",
            "accelerate",
            "spaces",
            "fastembed-gpu",
            "onnxruntime-gpu",
        }
        & indexed.keys()
    )
    assert str(indexed["fastembed"].specifier) == "==0.8.0"
    assert {"gradio", "httpx", "pillow", "numpy", "pydantic-settings"} <= indexed.keys()
    assert "python_version: 3.12.12" in (ROOT / "deploy/zerogpu/README.md").read_text()
    assert "suggested_hardware: cpu-basic" in (ROOT / "deploy/zerogpu/README.md").read_text()
