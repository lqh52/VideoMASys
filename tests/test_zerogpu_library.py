"""Bounded library-thumbnail validation and native Gradio selection contracts."""

import base64
import io
import json

import pytest

Image = pytest.importorskip("PIL.Image")

from videomasys.zerogpu.library import (  # noqa: E402
    MAX_THUMBNAIL_BYTES,
    thumbnail_css,
    thumbnail_data_uri,
)


def write_picture(path, image_format="PNG", size=(64, 36)):
    with Image.new("RGB", size, "blue") as picture:
        picture.save(path, format=image_format)
    return path


@pytest.mark.parametrize(
    ("image_format", "filename", "mime"),
    [
        ("JPEG", "poster.jpg", "image/jpeg"),
        ("JPEG", "poster.JPEG", "image/jpeg"),
        ("PNG", "poster.png", "image/png"),
        ("PNG", "poster.jpg", "image/png"),
    ],
)
def test_thumbnail_data_uri_preserves_valid_bounded_raster(tmp_path, image_format, filename, mime):
    path = write_picture(tmp_path / filename, image_format)
    uri = thumbnail_data_uri(tmp_path, filename)
    prefix, encoded = uri.split(",", 1)
    assert prefix == f"data:{mime};base64"
    data = base64.b64decode(encoded, validate=True)
    assert data == path.read_bytes()
    assert 0 < len(data) <= MAX_THUMBNAIL_BYTES
    with Image.open(io.BytesIO(data)) as picture:
        assert picture.format == image_format
        assert picture.size == (64, 36)


def test_thumbnail_accepts_nested_path_and_exact_dimension_limit(tmp_path):
    nested = tmp_path / "previews"
    nested.mkdir()
    write_picture(nested / "poster.png", size=(1024, 1024))
    assert thumbnail_data_uri(tmp_path, "previews/poster.png").startswith("data:image/png;base64,")


@pytest.mark.parametrize("relative", [None, "", 3, [], {}])
def test_thumbnail_rejects_non_string_or_empty_paths(tmp_path, relative):
    with pytest.raises(ValueError, match="relative file path"):
        thumbnail_data_uri(tmp_path, relative)


def test_thumbnail_rejects_absolute_and_parent_traversal_paths(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    outside = write_picture(tmp_path / "outside.png")
    inside = write_picture(root / "inside.png")
    for relative in (str(outside), str(inside), "../outside.png", "../library/inside.png"):
        with pytest.raises(ValueError, match="stay inside"):
            thumbnail_data_uri(root, relative)


@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("symlink_parent", [False, True])
def test_thumbnail_rejects_file_and_parent_symlinks_even_to_internal_targets(
    tmp_path, outside, symlink_parent
):
    root = tmp_path / "library"
    root.mkdir()
    target_root = tmp_path / "outside" if outside else root / "actual"
    target_root.mkdir()
    target = write_picture(target_root / "poster.png")
    if symlink_parent:
        (root / "previews").symlink_to(target_root, target_is_directory=True)
        relative = "previews/poster.png"
    else:
        (root / "poster.png").symlink_to(target)
        relative = "poster.png"
    with pytest.raises(ValueError, match="no symlinks"):
        thumbnail_data_uri(root, relative)


@pytest.mark.parametrize("size", [0, MAX_THUMBNAIL_BYTES + 1])
def test_thumbnail_rejects_invalid_byte_size_before_decoding(tmp_path, monkeypatch, size):
    (tmp_path / "poster.png").write_bytes(b"x" * size)

    def must_not_decode(*args, **kwargs):
        raise AssertionError("Out-of-bounds thumbnail must not reach image decoding")

    monkeypatch.setattr(Image, "open", must_not_decode)
    with pytest.raises(ValueError, match="at most 64 KiB"):
        thumbnail_data_uri(tmp_path, "poster.png")


def test_thumbnail_rejects_missing_nonimage_disallowed_format_and_large_dimensions(tmp_path):
    with pytest.raises(ValueError, match="existing JPEG or PNG"):
        thumbnail_data_uri(tmp_path, "missing.png")
    (tmp_path / "text.png").write_text("not an image")
    with pytest.raises(ValueError, match="Invalid library thumbnail"):
        thumbnail_data_uri(tmp_path, "text.png")
    write_picture(tmp_path / "disguised.png", image_format="GIF")
    with pytest.raises(ValueError, match="JPEG or PNG"):
        thumbnail_data_uri(tmp_path, "disguised.png")
    write_picture(tmp_path / "actual.gif", image_format="GIF")
    with pytest.raises(ValueError, match="existing JPEG or PNG"):
        thumbnail_data_uri(tmp_path, "actual.gif")
    write_picture(tmp_path / "wide.png", size=(1025, 2))
    with pytest.raises(ValueError, match="1024 pixels"):
        thumbnail_data_uri(tmp_path, "wide.png")


def test_thumbnail_rejects_corrupted_png(tmp_path):
    path = write_picture(tmp_path / "truncated.png")
    path.write_bytes(path.read_bytes()[:-12])
    with pytest.raises(ValueError, match="Invalid library thumbnail"):
        thumbnail_data_uri(tmp_path, path.name)


def test_optional_thumbnail_preserves_existing_library_and_css_positions(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("numpy")
    pytest.importorskip("gradio")
    from videomasys.zerogpu.ui import read_library

    assert read_library(tmp_path) == []
    (tmp_path / "video.mp4").write_bytes(b"isolated video-path fixture")
    write_picture(tmp_path / "poster.png")
    (tmp_path / "library.json").write_text(
        json.dumps(
            [
                {"title": "No preview", "file": "video.mp4", "question": "First question?"},
                {
                    "title": "With preview",
                    "file": "video.mp4",
                    "question": "Second question?",
                    "thumbnail": "poster.png",
                },
            ]
        )
    )
    library = read_library(tmp_path)
    assert library[0]["path"] == str(tmp_path / "video.mp4")
    assert library[0]["question"] == "First question?"
    assert library[0]["thumbnail_data"] is None
    assert library[1]["thumbnail_data"] == thumbnail_data_uri(tmp_path, "poster.png")
    css = thumbnail_css(library)
    assert "label:nth-child(2)" in css
    assert "label:nth-child(1)" not in css
    assert library[1]["thumbnail_data"] in css
    assert str(tmp_path) not in css
    assert thumbnail_css([library[0]]) == ""


def test_thumbnail_ui_retains_native_radio_and_adds_no_analysis_events(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("numpy")
    pytest.importorskip("gradio")
    from videomasys.zerogpu.ui import build_demo

    (tmp_path / "video.mp4").write_bytes(b"isolated video-path fixture")
    write_picture(tmp_path / "poster.jpg", image_format="JPEG")
    item = {"title": "Example", "file": "video.mp4", "question": "What happened?"}
    manifest = tmp_path / "library.json"
    manifest.write_text(json.dumps([item]))
    baseline = build_demo(cache_root=tmp_path / "cache", library_root=tmp_path)
    baseline_config = baseline.get_config_file()
    baseline.close()
    manifest.write_text(json.dumps([{**item, "thumbnail": "poster.jpg"}]))
    demo = build_demo(cache_root=tmp_path / "cache", library_root=tmp_path)
    try:
        config = demo.get_config_file()
        radios = [c for c in config["components"] if c["type"] == "radio"]
        assert len(radios) == 1
        radio = radios[0]
        assert radio["props"]["elem_id"] == "example-library"
        assert radio["props"]["type"] == "index"
        assert radio["props"].get("value") is None
        assert "with-thumbnails" in radio["props"]["elem_classes"]
        wrapper = next(
            c for c in config["components"] if c["props"].get("elem_id") == "example-cards"
        )
        assert wrapper["type"] == "html"
        assert wrapper["props"]["html_template"] == "@children"
        assert wrapper["props"]["apply_default_css"] is False
        assert wrapper["props"].get("js_on_load") is None
        assert thumbnail_data_uri(tmp_path, "poster.jpg") in wrapper["props"]["css_template"]
        assert "label:nth-child(1)" in wrapper["props"]["css_template"]
        assert [d["api_name"] for d in config["dependencies"]] == [
            d["api_name"] for d in baseline_config["dependencies"]
        ]
        analyze = [d for d in config["dependencies"] if d["api_name"] == "analyze"]
        assert len(analyze) == 1
        assert all(tuple(target) != (radio["id"], "change") for target in analyze[0]["targets"])
        choose = next(d for d in config["dependencies"] if d["api_name"] == "choose")
        assert (radio["id"], "change") in map(tuple, choose["targets"])
        assert demo.fns[choose["id"]].fn(0)[:3] == (
            str(tmp_path / "video.mp4"),
            "What happened?",
            "",
        )
    finally:
        demo.close()
