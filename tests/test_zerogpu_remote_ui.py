"""Offline integration checks: authored lifecycle fixtures, never model answers."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("cache_hit", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_remote_ui_admits_one_session_and_closes_it(tmp_path, monkeypatch, cache_hit, cancel):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    from videomasys.zerogpu import ui

    calls = []
    bundle = {"trace": [], "scenes": [], "database": str(tmp_path / "cache.sqlite3"),
              "cached": False, "duration": 60}
    monkeypatch.setattr(ui, "prepare_video", lambda *args: bundle)
    monkeypatch.setattr(ui, "SceneCache", lambda *args: SimpleNamespace(
        load=lambda *args: ([], []) if cache_hit else None,
    ))

    class Pipeline:
        signature = "test-remote-signature"
        model_id = "Qwen/Qwen3-VL-8B-Instruct:featherless-ai@test"

        @contextmanager
        def analysis_session(self, current):
            calls.append("admit")
            assert current is bundle
            try:
                yield self
            finally:
                calls.append("close")

        def index(self, current):
            calls.append("index")

        def save_index(self, current):
            calls.append("save")

        def qa(self, current, question, options):
            calls.append("qa")
            try:
                current["trace"].append({"stage": "Planner", "status": "complete"})
                yield current
                current["status"] = "needs_review"
            finally:
                calls.append("qa-close")

    def forbidden(*args):
        raise AssertionError("Remote UI must not use the local GPU callbacks")

    demo = ui.build_demo(Pipeline(), forbidden, forbidden,
                         cache_root=tmp_path, library_root=tmp_path / "none")
    run = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == "run")
    generator = run("fixture.mp4", "A fixture question?", "", None)
    try:
        while True:
            update = next(generator)
            if cancel and "qa" in calls:
                generator.close()
                break
            if 'data-state="complete"' in update[0]:
                assert calls[-2:] == ["qa-close", "close"]
    except StopIteration:
        pass
    assert calls.count("admit") == calls.count("close") == 1
    assert calls.count("index") == (0 if cache_hit else 1)
    assert calls.count("save") == (0 if cache_hit else 1)
    assert calls[-2:] == ["qa-close", "close"]
    demo.close()


def test_hosted_usage_is_in_raw_trace_without_changing_pipeline_steps():
    from videomasys.zerogpu import ui

    bundle = {"trace": [{"stage": "Planner", "status": "complete"}],
              "inference": {"backend": "hosted API", "calls": [{"image_count": 6}]}}
    _, _, trace = ui.display(bundle)
    assert trace[-1]["stage"] == "Hosted inference usage"
    assert trace[-1]["calls"][0]["image_count"] == 6
    assert len(bundle["trace"]) == 1


def test_openai_ui_names_actual_model_and_data_recipient(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    from videomasys.zerogpu import ui
    from videomasys.zerogpu.remote_runtime import RemotePipeline

    demo = ui.build_demo(
        RemotePipeline(provider="openai", token=""),
        cache_root=tmp_path, library_root=tmp_path / "none",
    )
    try:
        html_values = "\n".join(
            str(component.get("props", {}).get("value", ""))
            for component in demo.config["components"] if component["type"] == "html"
        )
        assert "gpt-5.6-luna" in html_values
        assert "sampled frames and your question to OpenAI." in html_values
        assert "Featherless" not in html_values
    finally:
        demo.close()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_hosted_failure_names_stage_and_closes_before_final_view(
    tmp_path, monkeypatch, cleanup_failure,
):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    from videomasys.zerogpu import ui
    from videomasys.zerogpu.remote import HostedVLMError

    calls = []
    bundle = {"trace": [], "scenes": [], "database": str(tmp_path / "cache.sqlite3"),
              "cached": False, "duration": 60}
    monkeypatch.setattr(ui, "prepare_video", lambda *args: bundle)
    monkeypatch.setattr(ui, "SceneCache", lambda *args: SimpleNamespace(load=lambda *args: None))

    class Pipeline:
        signature = "test-remote-signature"
        model_id = "test-only-model"

        @contextmanager
        def analysis_session(self, current):
            calls.append("admit")
            try:
                yield self
            finally:
                calls.append("close")
                current["inference"] = {"calls": [{"error_code": "invalid_response"}],
                                        "allowance": {"request_attempts_used": 1}}
                if cleanup_failure:
                    raise RuntimeError("test-only sensitive cleanup exception")

        def index(self, current):
            current["trace"].append({"stage": "Caption scene 0", "status": "failed"})
            raise HostedVLMError("invalid_response")

    demo = ui.build_demo(Pipeline(), cache_root=tmp_path, library_root=tmp_path / "none")
    run = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == "run")
    generator = run("fixture.mp4", "A fixture question?", "", None)
    try:
        for update in generator:
            if 'data-state="error"' not in update[0]:
                continue
            assert calls == ["admit", "close"]
            assert "Caption scene 0:" in update[0]
            assert not bundle.get("answer")
            assert bundle["trace"][-1]["failed_stage"] == "Caption scene 0"
            assert bool(bundle["trace"][-1].get("cleanup_warning")) == cleanup_failure
            assert "sensitive cleanup exception" not in str(update)
            assert update[3][-1]["stage"] == "Hosted inference usage"
            assert update[3][-1]["allowance"]["request_attempts_used"] == 1
            break
        else:
            pytest.fail("Expected a visible failure")
    finally:
        generator.close()
        demo.close()
    assert calls.count("close") == 1
