"""Exercise hosted inference with synthetic media, never benchmark/private videos."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from gradio_client import Client, handle_file
from huggingface_hub import get_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space")
    parser.add_argument("--repeat", action="store_true", help="Also verify a cache-hit run")
    args = parser.parse_args()
    client = Client(args.space, token=get_token(), download_files=False, verbose=False)
    with tempfile.TemporaryDirectory(prefix="videomasys-smoke-") as directory:
        video = Path(directory) / "synthetic-blue.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=blue:s=160x120",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                str(video),
            ],
            check=True,
            timeout=30,
        )
        for attempt in range(2 if args.repeat else 1):
            job = client.submit(
                handle_file(str(video)),
                "What color fills the screen?",
                "",
                api_name="/analyze",
            )
            result = job.result(timeout=300)
            status, answer, _gallery, trace = result[:4]
            print(json.dumps({"run": attempt + 1, "status": status, "answer": answer}))
            print(json.dumps({"trace": trace}), flush=True)
            if not answer or trace[-1].get("stage") != "Verifier":
                raise SystemExit(
                    "Hosted pipeline did not complete; inspect the actual trace above."
                )
            if attempt and not any(
                step.get("stage") == "Video cache" and step.get("cache_hit") for step in trace
            ):
                raise SystemExit("The repeated video did not reuse its scene cache.")


if __name__ == "__main__":
    main()
