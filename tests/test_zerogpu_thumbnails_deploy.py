"""The Space publishes only explicitly selected, bounded local image assets."""

import importlib.util
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def deployment():
    source = Path(__file__).parents[1] / "scripts/deploy_zerogpu.py"
    spec = importlib.util.spec_from_file_location("thumbnail_deployment", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library(root, thumbnail=None):
    (root / "example.mp4").write_bytes(b"isolated video fixture")
    entry = {"title": "Example", "file": "example.mp4", "question": "What happened?"}
    if thumbnail is not None:
        entry["thumbnail"] = thumbnail
    (root / "library.json").write_text(json.dumps([entry]))


def test_only_manifest_referenced_thumbnail_is_published(tmp_path, deployment):
    folder = tmp_path / "thumbnails"
    folder.mkdir()
    Image.new("RGB", (384, 216), "teal").save(folder / "selected.jpg")
    Image.new("RGB", (384, 216), "red").save(folder / "not-selected.png")
    library(tmp_path, "thumbnails/selected.jpg")
    files = deployment.deployment_files(tmp_path)
    assert files["space_library/thumbnails/selected.jpg"] == folder / "selected.jpg"
    assert "space_library/thumbnails/not-selected.png" not in files
    assert not any("thumbnail" in name for name in deployment.deployment_files())


def test_thumbnail_is_optional_for_existing_libraries(tmp_path, deployment):
    library(tmp_path)
    files = deployment.deployment_files(tmp_path)
    assert "space_library/example.mp4" in files
    assert not any(name.endswith((".jpg", ".png")) for name in files)


@pytest.mark.parametrize("thumbnail", ["../outside.jpg", "/tmp/outside.jpg"])
def test_thumbnail_path_must_stay_in_library(tmp_path, deployment, thumbnail):
    library(tmp_path, thumbnail)
    with pytest.raises(ValueError, match="inside"):
        deployment.deployment_files(tmp_path)


@pytest.mark.parametrize("symlink_kind", ["file", "folder"])
def test_thumbnail_and_parent_symlinks_are_rejected(tmp_path, deployment, symlink_kind):
    folder = tmp_path / "real"
    folder.mkdir()
    Image.new("RGB", (20, 20), "teal").save(folder / "image.jpg")
    if symlink_kind == "file":
        (tmp_path / "linked.jpg").symlink_to(folder / "image.jpg")
        thumbnail = "linked.jpg"
    else:
        (tmp_path / "linked").symlink_to(folder, target_is_directory=True)
        thumbnail = "linked/image.jpg"
    library(tmp_path, thumbnail)
    with pytest.raises(ValueError, match="symlink"):
        deployment.deployment_files(tmp_path)


@pytest.mark.parametrize("suffix", [".svg", ".html", ".gif"])
def test_only_jpeg_and_png_thumbnails_allowed(tmp_path, deployment, suffix):
    thumbnail = "image" + suffix
    (tmp_path / thumbnail).write_bytes(b"not a permitted image")
    library(tmp_path, thumbnail)
    with pytest.raises(ValueError, match="JPEG or PNG"):
        deployment.deployment_files(tmp_path)


@pytest.mark.parametrize("size", [0, 64 * 1024 + 1])
def test_thumbnail_file_size_is_bounded(tmp_path, deployment, size):
    (tmp_path / "image.jpg").write_bytes(b"x" * size)
    library(tmp_path, "image.jpg")
    with pytest.raises(ValueError, match="64 KiB"):
        deployment.deployment_files(tmp_path)


def test_invalid_jpeg_content_is_rejected(tmp_path, deployment):
    (tmp_path / "image.jpg").write_bytes(b"not an image")
    library(tmp_path, "image.jpg")
    with pytest.raises(ValueError, match="valid JPEG or PNG"):
        deployment.deployment_files(tmp_path)


def test_compressed_thumbnail_dimensions_are_bounded(tmp_path, deployment):
    Image.new("RGB", (2048, 1), "teal").save(tmp_path / "image.png")
    library(tmp_path, "image.png")
    with pytest.raises(ValueError, match="1024"):
        deployment.deployment_files(tmp_path)


def test_curated_thumbnails_are_small_and_deployable(deployment):
    root = Path(__file__).parents[1] / "space_library"
    entries = json.loads((root / "library.json").read_text())
    files = deployment.deployment_files(root)
    thumbnails = [item["thumbnail"] for item in entries]
    assert len(thumbnails) == 4
    assert len(set(thumbnails)) == 4
    for thumbnail in thumbnails:
        path = files[f"space_library/{thumbnail}"]
        assert path.stat().st_size <= 40 * 1024
        with Image.open(path) as image:
            assert image.format == "JPEG"
            assert image.size == (384, 216)
