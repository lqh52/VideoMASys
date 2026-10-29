"""CPU orchestration with lazy ONNX embeddings and an explicitly routed hosted VLM.

Importing or building the UI makes no model request and loads no GPU libraries.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager

import numpy as np

from videomasys.zerogpu.core import LivePipeline
from videomasys.zerogpu.usage import UsageGuard

MODEL = "Qwen/Qwen3-VL-8B-Instruct:featherless-ai"
OPENAI_MODEL = "gpt-5.6-luna"
GPT41_MINI_MODEL = "gpt-4.1-mini-2025-04-14"
GPT54_MINI_MODEL = "gpt-5.4-mini-2026-03-17"
GPT54_MODEL = "gpt-5.4-2026-03-05"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Separate cache identity: ONNX vectors must not silently mix with old Torch vectors.
EMBEDDING_ID = f"{EMBEDDING_MODEL}@fastembed-0.8.0-onnx-v1"


class CPUEmbedder:
    def __init__(self):
        self.encoder = None
        self.lock = threading.Lock()

    def __call__(self, texts):
        if len(texts) > 64 or any(len(text) > 20000 for text in texts):
            raise RuntimeError("Embedding input exceeds the demo limit.")
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with self.lock:
            if self.encoder is None:
                try:
                    from fastembed import TextEmbedding

                    self.encoder = TextEmbedding(
                        model_name=EMBEDDING_MODEL,
                        threads=2,
                        providers=["CPUExecutionProvider"],
                        cache_dir=os.environ.get("EMBEDDING_CACHE_DIR"),
                    )
                except Exception:
                    raise RuntimeError(
                        "The CPU search model is unavailable. Please try again later."
                    ) from None
            vectors = np.asarray(list(self.encoder.embed(texts)), dtype=np.float32)
        if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
            raise RuntimeError("The CPU search model returned invalid embeddings.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("The CPU search model returned invalid embeddings.")
        return vectors / norms


class RemotePipeline(LivePipeline):
    def __init__(self, *, provider="api", token=None, guard=None, embed=None, vlm_factory=None,
                 prompt_profile=None):
        if provider not in {"api", "openai"}:
            raise ValueError("Hosted provider must be 'api' or 'openai'.")
        self.provider = provider
        self._secret_name = "OPENAI_API_KEY" if provider == "openai" else "HF_INFERENCE_TOKEN"
        self._allowed_model = OPENAI_MODEL if provider == "openai" else MODEL
        self._token = token if token is not None else os.environ.get(self._secret_name, "")
        self._model = os.environ.get("VLM_MODEL", self._allowed_model)
        self.provider_label = "OpenAI" if provider == "openai" else "Hugging Face and Featherless"
        self._guard = guard
        self._guard_lock = threading.Lock()
        self._factory = vlm_factory
        # Hosted providers do not guarantee our old exact local weights revision.
        super().__init__(
            self._outside_session,
            embed or CPUEmbedder(),
            f"{self._model}@openai-chat-v1-none-strict"
            if provider == "openai" else f"{self._model}@hosted-api-v2-rp108",
            EMBEDDING_ID,
            prompt_profile=(
                os.environ.get("VLM_PROMPT_PROFILE", "clean")
                if prompt_profile is None else prompt_profile
            ),
        )

    def _usage_guard(self):
        with self._guard_lock:
            if self._guard is None:
                try:
                    self._guard = UsageGuard(
                        os.environ.get("DEMO_USAGE_DB", "/tmp/videomasys-usage.sqlite3"),
                        max_runs=int(os.environ.get("DEMO_MAX_RUNS", "5")),
                        max_requests=int(os.environ.get("DEMO_MAX_REQUESTS", "120")),
                        max_requests_per_run=int(os.environ.get("DEMO_MAX_REQUESTS_PER_RUN", "32")),
                    )
                except ValueError:
                    raise RuntimeError(
                        "Demo usage limits are misconfigured. Ask the owner to review them."
                    ) from None
        return self._guard

    @contextmanager
    def analysis_session(self, bundle):
        if not self._token or not self._token.strip():
            raise RuntimeError(
                "Live analysis is not configured yet. "
                f"The Space owner needs to add the {self._secret_name} secret."
            )
        allowed_models = (
            {OPENAI_MODEL, GPT41_MINI_MODEL, GPT54_MINI_MODEL, GPT54_MODEL}
            if self.provider == "openai" else {MODEL}
        )
        if self._model not in allowed_models:
            if self.provider == "openai":
                raise RuntimeError(
                    "The OpenAI backend allows only the explicitly configured models."
                )
            raise RuntimeError("This demo is configured for Qwen3-VL-8B on Featherless only.")
        guard = self._usage_guard()
        allowance = guard.start_run()
        vlm = None
        try:
            factory = self._factory
            if factory is None:
                if self.provider == "openai":
                    from videomasys.zerogpu.openai import (
                        OpenAI41MiniVLM,
                        OpenAI54MiniVLM,
                        OpenAI54VLM,
                        OpenAIVLM,
                    )

                    factory = {
                        OPENAI_MODEL: OpenAIVLM,
                        GPT41_MINI_MODEL: OpenAI41MiniVLM,
                        GPT54_MINI_MODEL: OpenAI54MiniVLM,
                        GPT54_MODEL: OpenAI54VLM,
                    }[self._model]
                else:
                    from videomasys.zerogpu.remote import HostedVLM

                    factory = HostedVLM
            # Only the opt-in evidence graph requires identical encoded pixels
            # for per-window inspection and the larger answer/review image pack.
            image_options = (
                {"image_bounds": (512, 384)} if self.prompt_profile == "structured" else {}
            )
            vlm = factory(self._token, model=self._model,
                          before_request=allowance.before_request, **image_options)
            bundle["inference"] = {
                "backend": "hosted API",
                "provider": self.provider_label,
                "model": self._model,
                "prompt_profile": self.prompt_profile,
                "prompt_version": self.prompt_version,
                "weights_revision": (
                    "pinned API snapshot: 2025-04-14; serving implementation provider-managed"
                    if self.provider == "openai" and self._model == GPT41_MINI_MODEL
                    else "pinned API snapshot: 2026-03-17; serving implementation provider-managed"
                    if self.provider == "openai" and self._model == GPT54_MINI_MODEL
                    else "pinned API snapshot: 2026-03-05; serving implementation provider-managed"
                    if self.provider == "openai" and self._model == GPT54_MODEL
                    else "provider-managed; not pinned"
                ),
                "calls": [],
                "allowance": guard.snapshot(),
            }

            def generate(prompt, paths, tokens):
                try:
                    return vlm(prompt, paths, tokens)
                finally:
                    bundle["inference"]["calls"] = list(vlm.calls)
                    self._record_usage(bundle, guard)

            def generate_structured(prompt, paths, tokens, schema):
                try:
                    return vlm.generate_structured(prompt, paths, tokens, schema)
                finally:
                    bundle["inference"]["calls"] = list(vlm.calls)
                    self._record_usage(bundle, guard)

            # Gradio may advance a sync generator on different worker threads.
            # Bind credentials/allowance to this run, not ambient thread context.
            yield LivePipeline(
                generate, self.embed, self.model_id, self.embedding_id,
                generate_structured=generate_structured if self.provider == "openai" else None,
                prompt_profile=self.prompt_profile,
            )
        finally:
            allowance.close()
            try:
                self._record_usage(bundle, guard)
            finally:
                if vlm is not None and hasattr(vlm, "close"):
                    vlm.close()

    @staticmethod
    def _record_usage(bundle, guard):
        if "inference" in bundle:
            try:
                bundle["inference"]["allowance"] = guard.snapshot()
            except sqlite3.Error:
                # A display-only read must not hide the original request error.
                # Request admission/reservation still fails closed on ledger errors.
                bundle["inference"]["allowance"] = {"status": "Usage summary unavailable"}

    @staticmethod
    def _outside_session(prompt, paths, tokens):
        raise RuntimeError("Hosted inference requires an admitted analysis session.")


def load_runtime(*, provider="api"):
    pipeline = RemotePipeline(provider=provider)

    def index(bundle):
        try:
            pipeline.index(bundle)
            pipeline.save_index(bundle)
        except Exception as exc:
            bundle["error"] = str(exc)
        return bundle

    def qa(bundle, question, options):
        try:
            yield from pipeline.qa(bundle, question, options)
        except Exception as exc:
            bundle["error"] = str(exc)
            yield bundle

    return pipeline, index, qa
