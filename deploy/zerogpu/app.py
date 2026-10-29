"""CPU/API Space entrypoint; local GPU inference is an explicit optional backend."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

os.environ.setdefault("GRADIO_TEMP_DIR", f"/tmp/videomasys-gradio-{os.getuid()}")

preview = os.environ.get("VIDEOMASYS_UI_ONLY") == "1"
backend = os.environ.get("VLM_BACKEND", "api").strip().lower()
if not preview and backend not in {"api", "openai", "local"}:
    raise ValueError("VLM_BACKEND must be 'api', 'openai', or 'local'.")

if not preview and backend == "local":
    import spaces  # noqa: F401 -- installs CUDA emulation before other libraries

from videomasys.zerogpu.ui import STYLE, THEME, build_demo  # noqa: E402 -- backend setup first

if preview:
    demo = build_demo()
elif backend == "local":
    from videomasys.zerogpu.inference import load_runtime

    demo = build_demo(*load_runtime())
else:
    from videomasys.zerogpu.remote_runtime import load_runtime

    runtime = load_runtime(provider="openai") if backend == "openai" else load_runtime()
    demo = build_demo(*runtime)

if __name__ == "__main__":
    demo.launch(
        css=STYLE,
        theme=THEME,
        server_name="0.0.0.0",
        max_file_size="100mb",
        show_error=True,
        allowed_paths=["space_library"],
    )
