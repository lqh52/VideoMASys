"""Small, owner-provided library previews; never decoded on the analysis path."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_THUMBNAIL_BYTES = 64 * 1024


def thumbnail_data_uri(root: Path, relative: str) -> str:
    """Embed a bounded raster asset without extra authenticated media requests."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Library thumbnails need a relative file path.")
    relative_path = Path(relative)
    root = root.resolve()
    path = root / relative_path
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or not path.resolve().is_relative_to(root)
        or any(
            (root / Path(*relative_path.parts[:index])).is_symlink()
            for index in range(1, len(relative_path.parts) + 1)
        )
    ):
        raise ValueError("Library thumbnails must stay inside the library folder; no symlinks.")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
        raise ValueError("Library thumbnails must be existing JPEG or PNG files.")
    if not 0 < path.stat().st_size <= MAX_THUMBNAIL_BYTES:
        raise ValueError("Library thumbnails must be at most 64 KiB.")
    data = path.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as picture:
            if picture.format not in {"JPEG", "PNG"} or max(picture.size) > 1024:
                raise ValueError("Use a JPEG or PNG thumbnail no larger than 1024 pixels.")
            mime = "image/jpeg" if picture.format == "JPEG" else "image/png"
            picture.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Invalid library thumbnail image.") from exc
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def thumbnail_css(library: list[dict]) -> str:
    """Only numeric selectors and validated raster data enter these scoped rules."""
    return "\n".join(
        f'#example-library .wrap > label:nth-child({index + 1}) '
        f'{{--example-image:url("{item["thumbnail_data"]}");}}'
        for index, item in enumerate(library)
        if item.get("thumbnail_data")
    )
