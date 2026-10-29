"""Opt-in local memory preflight, not a model accuracy or L4 latency benchmark.

Uses synthetic images and fills the existing 8192-token context budget. Select
an unused local GPU with CUDA_VISIBLE_DEVICES; no remote hardware is requested.
"""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator-gib", type=float, default=22.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    total = torch.cuda.get_device_properties(0).total_memory
    if not 0 < args.allocator_gib * 1024**3 <= total:
        parser.error("The allocator limit must fit the selected local GPU.")
    torch.cuda.set_per_process_memory_fraction(args.allocator_gib * 1024**3 / total)
    torch.set_num_threads(2)
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    images = [Image.new("RGB", (512, 384), (i * 12, 80, 120)) for i in range(18)]

    def prepare(words):
        return processor.apply_chat_template(
            [{"role": "user", "content": [
                *[{"type": "image", "image": picture} for picture in images],
                {"type": "text", "text": "Describe the images. " + "video " * words},
            ]}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )

    low, high = 0, 8192
    while low < high:
        mid = (low + high + 1) // 2
        if prepare(mid)["input_ids"].shape[-1] <= 8192 - 750:
            low = mid
        else:
            high = mid - 1
    inputs = prepare(low)
    inputs.pop("token_type_ids", None)
    prompt_tokens = inputs["input_ids"].shape[-1]
    if prompt_tokens > 8192 - 750:
        raise RuntimeError("Images alone exceeded the app context budget.")
    inputs = inputs.to("cuda")
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=750, min_new_tokens=750,
            do_sample=False, repetition_penalty=1.08, use_cache=True,
        )
    torch.cuda.synchronize()
    report = {
        "purpose": "synthetic memory preflight only; not L4 verification",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "model": model_id, "revision": model.config._commit_hash,
        "allocator_limit_gib": args.allocator_gib,
        "images": len(images), "prompt_tokens": prompt_tokens,
        "generated_tokens": output.shape[-1] - prompt_tokens,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "generation_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
