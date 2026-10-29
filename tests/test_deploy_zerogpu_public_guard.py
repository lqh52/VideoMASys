"""Public-target opt-in is explicit; mocked deployments never contact Hugging Face."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    source = Path(__file__).parents[1] / "scripts/deploy_zerogpu.py"
    spec = importlib.util.spec_from_file_location("public_guard_deployment", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    asset = tmp_path / "app.py"
    asset.write_text("# Isolated deployment fixture; never uploaded.\n")
    monkeypatch.setattr(module, "deployment_files", lambda _library: {"app.py": asset})
    # Visibility-changing methods are deliberately absent: an attempted call to
    # one fails the test, instead of being silently accepted by an unrestricted mock.
    api = Mock(
        spec_set=[
            "create_repo",
            "repo_info",
            "create_commit",
            "get_space_runtime",
            "request_space_hardware",
        ]
    )
    api.repo_info.return_value = SimpleNamespace(private=True)
    api.create_commit.return_value = SimpleNamespace(oid="isolated-test-commit")
    api.get_space_runtime.return_value = SimpleNamespace(
        requested_hardware=module.SpaceHardware.ZERO_A10G
    )
    factory = Mock(return_value=api)
    monkeypatch.setattr(module, "HfApi", factory)
    return SimpleNamespace(module=module, api=api, factory=factory, asset=asset)


def run(deployment, monkeypatch, *flags):
    monkeypatch.setattr(sys, "argv", ["deploy_zerogpu.py", "test-owner/test-space", *flags])
    deployment.module.main()


def assert_private_creation_only(deployment):
    deployment.api.create_repo.assert_called_once_with(
        "test-owner/test-space",
        repo_type="space",
        private=True,
        space_sdk="gradio",
        space_hardware=deployment.module.SpaceHardware.CPU_BASIC,
        exist_ok=False,
    )


def test_existing_public_space_is_refused_without_opt_in(deployment, monkeypatch):
    deployment.api.repo_info.return_value = SimpleNamespace(private=False)
    with pytest.raises(RuntimeError, match="--allow-public"):
        run(deployment, monkeypatch)
    deployment.api.create_repo.assert_not_called()
    deployment.api.repo_info.assert_called_once_with("test-owner/test-space", repo_type="space")
    deployment.api.create_commit.assert_not_called()
    deployment.api.get_space_runtime.assert_not_called()
    deployment.api.request_space_hardware.assert_not_called()


@pytest.mark.parametrize("private", [True, False])
def test_opt_in_allows_upload_without_changing_visibility(deployment, monkeypatch, private):
    info = SimpleNamespace(private=private)
    deployment.api.repo_info.return_value = info
    run(deployment, monkeypatch, "--allow-public")
    deployment.api.create_repo.assert_not_called()
    assert info.private is private
    deployment.api.create_commit.assert_called_once()
    args, kwargs = deployment.api.create_commit.call_args
    assert args == ("test-owner/test-space",)
    assert kwargs["repo_type"] == "space"
    assert len(kwargs["operations"]) == 1
    assert kwargs["operations"][0].path_in_repo == "app.py"
    assert Path(kwargs["operations"][0].path_or_fileobj) == deployment.asset
    deployment.api.request_space_hardware.assert_not_called()


def test_private_space_still_updates_without_opt_in(deployment, monkeypatch):
    run(deployment, monkeypatch)
    deployment.api.create_repo.assert_not_called()
    deployment.api.create_commit.assert_called_once()


@pytest.mark.parametrize("allow_public", [False, True])
def test_new_space_is_created_private_even_with_opt_in(deployment, monkeypatch, allow_public):
    state = {"exists": False, "private": None}

    def create_repo(_repo_id, **kwargs):
        assert not state["exists"]
        state.update(exists=True, private=kwargs["private"])

    def repo_info(_repo_id, **_kwargs):
        if not state["exists"]:
            raise missing_repo(deployment)
        return SimpleNamespace(private=state["private"])

    deployment.api.create_repo.side_effect = create_repo
    deployment.api.repo_info.side_effect = repo_info
    run(deployment, monkeypatch, *(["--allow-public"] if allow_public else []))
    assert_private_creation_only(deployment)
    assert state == {"exists": True, "private": True}
    deployment.api.create_commit.assert_called_once()


@pytest.mark.parametrize("allow_public", [False, True])
def test_dry_run_never_constructs_hf_api(deployment, monkeypatch, capsys, allow_public):
    flags = ["--dry-run", *(["--allow-public"] if allow_public else [])]
    run(deployment, monkeypatch, *flags)
    assert capsys.readouterr().out == "app.py\n"
    deployment.factory.assert_not_called()
    assert not deployment.api.mock_calls


@pytest.mark.parametrize("stage", ["create_repo", "repo_info"])
def test_opt_in_does_not_bypass_api_failures(deployment, monkeypatch, stage):
    if stage == "create_repo":
        deployment.api.repo_info.side_effect = missing_repo(deployment)
    getattr(deployment.api, stage).side_effect = RuntimeError("Isolated API failure")
    with pytest.raises(RuntimeError, match="Isolated API failure"):
        run(deployment, monkeypatch, "--allow-public")
    deployment.api.create_commit.assert_not_called()
    deployment.api.request_space_hardware.assert_not_called()


@pytest.mark.parametrize("hardware", [None, "cpu-basic", "zero-a10g", "l4x1", "l40s"])
def test_default_deploy_preserves_existing_hardware_and_sleep(deployment, monkeypatch, hardware):
    deployment.api.repo_info.return_value = SimpleNamespace(private=False)
    deployment.api.get_space_runtime.return_value = SimpleNamespace(
        requested_hardware=hardware, sleep_time=600
    )
    run(deployment, monkeypatch, "--allow-public")
    deployment.api.create_repo.assert_not_called()
    deployment.api.create_commit.assert_called_once()
    deployment.api.request_space_hardware.assert_not_called()


def missing_repo(deployment, status=404):
    return deployment.module.RepositoryNotFoundError(
        "Isolated missing repository",
        response=httpx.Response(status, request=httpx.Request("GET", "https://example.invalid")),
    )


@pytest.mark.parametrize("status", [401, 403])
def test_lookup_authentication_failures_never_create_or_mutate(deployment, monkeypatch, status):
    deployment.api.repo_info.side_effect = missing_repo(deployment, status)
    with pytest.raises(deployment.module.RepositoryNotFoundError):
        run(deployment, monkeypatch, "--allow-public")
    deployment.api.create_repo.assert_not_called()
    deployment.api.create_commit.assert_not_called()
    deployment.api.request_space_hardware.assert_not_called()


@pytest.mark.parametrize(
    "flags",
    [
        ["--hardware", "l4x1"],
        ["--hardware", "l4x1", "--sleep-seconds", "-1"],
        ["--hardware", "l4x1", "--sleep-seconds", "0"],
        ["--hardware", "l4x1", "--sleep-seconds", "-30"],
        ["--hardware", "l4x1", "--sleep-seconds", "not-an-integer"],
        ["--sleep-seconds", "600"],
        ["--hardware", "zero-a10g", "--sleep-seconds", "600"],
        ["--hardware", "cpu-basic", "--sleep-seconds", "600"],
        ["--hardware", "l40s", "--sleep-seconds", "600"],
    ],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_invalid_hardware_flags_fail_before_api_construction(
    deployment, monkeypatch, flags, dry_run
):
    with pytest.raises(SystemExit) as error:
        run(deployment, monkeypatch, *flags, *(["--dry-run"] if dry_run else []))
    assert error.value.code == 2
    deployment.factory.assert_not_called()
    assert not deployment.api.mock_calls


@pytest.mark.parametrize("private", [True, False])
def test_explicit_paid_request_sets_hardware_and_sleep_in_one_call(
    deployment, monkeypatch, private
):
    deployment.api.repo_info.return_value = SimpleNamespace(private=private)
    flags = [] if private else ["--allow-public"]
    run(deployment, monkeypatch, "--hardware", "l4x1", "--sleep-seconds", "600", *flags)
    deployment.api.create_repo.assert_not_called()
    deployment.api.create_commit.assert_called_once()
    deployment.api.request_space_hardware.assert_called_once_with(
        "test-owner/test-space", hardware=deployment.module.SpaceHardware.L4X1, sleep_time=600
    )
    calls = [call[0] for call in deployment.api.mock_calls]
    assert calls.index("create_commit") < calls.index("request_space_hardware")


def test_paid_public_request_requires_public_opt_in(deployment, monkeypatch):
    deployment.api.repo_info.return_value = SimpleNamespace(private=False)
    with pytest.raises(RuntimeError, match="--allow-public"):
        run(deployment, monkeypatch, "--hardware", "l4x1", "--sleep-seconds", "600")
    deployment.api.create_repo.assert_not_called()
    deployment.api.create_commit.assert_not_called()
    deployment.api.request_space_hardware.assert_not_called()


def test_new_paid_space_creation_includes_sleep_atomically(deployment, monkeypatch):
    deployment.api.repo_info.side_effect = [missing_repo(deployment), SimpleNamespace(private=True)]
    run(deployment, monkeypatch, "--hardware", "l4x1", "--sleep-seconds", "600")
    deployment.api.create_repo.assert_called_once_with(
        "test-owner/test-space",
        repo_type="space",
        private=True,
        space_sdk="gradio",
        space_hardware=deployment.module.SpaceHardware.L4X1,
        space_sleep_time=600,
        exist_ok=False,
    )
    deployment.api.request_space_hardware.assert_not_called()


def test_explicit_zero_request_is_not_an_automatic_fallback(deployment, monkeypatch):
    run(deployment, monkeypatch, "--hardware", "zero-a10g")
    deployment.api.request_space_hardware.assert_called_once_with(
        "test-owner/test-space", hardware=deployment.module.SpaceHardware.ZERO_A10G
    )


def test_paid_request_error_is_not_retried_or_replaced_by_other_hardware(deployment, monkeypatch):
    deployment.api.request_space_hardware.side_effect = RuntimeError("Isolated hardware failure")
    with pytest.raises(RuntimeError, match="Isolated hardware failure"):
        run(deployment, monkeypatch, "--hardware", "l4x1", "--sleep-seconds", "600")
    deployment.api.request_space_hardware.assert_called_once_with(
        "test-owner/test-space", hardware=deployment.module.SpaceHardware.L4X1, sleep_time=600
    )


def test_failed_upload_does_not_start_paid_hardware(deployment, monkeypatch):
    deployment.api.create_commit.side_effect = RuntimeError("Isolated upload failure")
    with pytest.raises(RuntimeError, match="Isolated upload failure"):
        run(deployment, monkeypatch, "--hardware", "l4x1", "--sleep-seconds", "600")
    deployment.api.request_space_hardware.assert_not_called()


def test_paid_dry_run_does_not_contact_api(deployment, monkeypatch):
    run(deployment, monkeypatch, "--hardware", "l4x1", "--sleep-seconds", "600", "--dry-run")
    deployment.factory.assert_not_called()
    assert not deployment.api.mock_calls


@pytest.mark.parametrize("private", [True, False])
def test_explicit_cpu_request_needs_no_sleep_and_preserves_visibility(
    deployment, monkeypatch, private
):
    deployment.api.repo_info.return_value = SimpleNamespace(private=private)
    run(
        deployment, monkeypatch, "--hardware", "cpu-basic", *([] if private else ["--allow-public"])
    )
    deployment.api.create_repo.assert_not_called()
    deployment.api.create_commit.assert_called_once()
    deployment.api.request_space_hardware.assert_called_once_with(
        "test-owner/test-space", hardware=deployment.module.SpaceHardware.CPU_BASIC
    )


def test_cpu_public_request_still_requires_explicit_public_opt_in(deployment, monkeypatch):
    deployment.api.repo_info.return_value = SimpleNamespace(private=False)
    with pytest.raises(RuntimeError, match="--allow-public"):
        run(deployment, monkeypatch, "--hardware", "cpu-basic")
    deployment.api.create_commit.assert_not_called()
    deployment.api.request_space_hardware.assert_not_called()


def test_new_explicit_cpu_space_is_private_with_no_custom_sleep(deployment, monkeypatch):
    deployment.api.repo_info.side_effect = [missing_repo(deployment), SimpleNamespace(private=True)]
    run(deployment, monkeypatch, "--hardware", "cpu-basic")
    assert_private_creation_only(deployment)
    deployment.api.request_space_hardware.assert_not_called()


def test_cpu_dry_run_does_not_contact_api(deployment, monkeypatch):
    run(deployment, monkeypatch, "--hardware", "cpu-basic", "--dry-run")
    deployment.factory.assert_not_called()
    assert not deployment.api.mock_calls


def test_cpu_request_failure_never_falls_back_to_paid_hardware(deployment, monkeypatch):
    deployment.api.request_space_hardware.side_effect = RuntimeError("Isolated CPU failure")
    with pytest.raises(RuntimeError, match="Isolated CPU failure"):
        run(deployment, monkeypatch, "--hardware", "cpu-basic")
    deployment.api.request_space_hardware.assert_called_once_with(
        "test-owner/test-space", hardware=deployment.module.SpaceHardware.CPU_BASIC
    )
