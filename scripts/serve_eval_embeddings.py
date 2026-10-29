"""Local-only, CPU MiniLM embeddings for reproducible video QA evaluations.

Run with the ML environment, for example:
  /vast/s224013995/videomasys-vllm/bin/python scripts/serve_eval_embeddings.py

Mean pooling, attention masking, normalization and the 256-token truncation limit
match sentence-transformers/all-MiniLM-L6-v2's SentenceTransformer configuration.
This is an evaluation helper, not a deployed or externally exposed model server.
"""

from __future__ import annotations

import argparse
import os

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def build_app(model_id=MODEL_ID):
    # Establish CPU-only behavior before importing any CUDA-aware library.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from fastapi import FastAPI, HTTPException
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=False).to("cpu").eval()
    app = FastAPI(title="VideoMASys evaluation embeddings")
    metadata = {
        "model": model_id,
        "revision": getattr(model.config, "_commit_hash", None),
        "device": "cpu",
        "dimensions": model.config.hidden_size,
        "pooling": "attention-mask-weighted mean",
        "normalize_embeddings": True,
        "max_seq_length": 256,
        "torch_threads": 2,
    }

    @app.get("/health")
    def health():
        return {"status": "ok", **metadata}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", **metadata}]}

    @app.post("/v1/embeddings")
    def embeddings(request: dict):
        if request.get("model", model_id) != model_id:
            raise HTTPException(400, "Unsupported embedding model.")
        texts = request.get("input")
        if isinstance(texts, str):
            texts = [texts]
        if (
            not isinstance(texts, list)
            or not 1 <= len(texts) <= 64
            or any(not isinstance(text, str) or len(text) > 20_000 for text in texts)
        ):
            raise HTTPException(400, "input must contain 1–64 strings of at most 20000 characters.")
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.inference_mode():
            token_vectors = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)
            pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).tolist()
        tokens = int(encoded["attention_mask"].sum().item())
        return {
            "object": "list",
            "model": model_id,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8206)
    args = parser.parse_args()
    import uvicorn

    app = build_app()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
