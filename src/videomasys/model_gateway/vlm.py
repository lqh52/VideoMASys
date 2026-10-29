from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel

from videomasys.settings import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelGatewayError(RuntimeError):
    pass


class VLMGateway:
    """One vLLM endpoint, with typed outputs and bounded role-specific calls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.vlm_max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=settings.vlm_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.vlm_api_key}"},
            timeout=httpx.Timeout(settings.vlm_timeout_seconds),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def complete_structured(
        self,
        *,
        role: str,
        prompt: str,
        response_model: type[SchemaT],
        image_paths: list[Path] | None = None,
        video_paths: list[Path] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> SchemaT:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for path in image_paths or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": await self._as_data_url(path)},
                }
            )
        for path in video_paths or []:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": await self._as_data_url(path)},
                }
            )

        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are the {role} in a video QA system. "
                        "Return only data matching the requested JSON schema. "
                        "Do not invent observations that are not in the supplied evidence."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "repetition_penalty": 1.05,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": self.settings.vlm_enable_thinking,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with self._semaphore:
                    response = await self._client.post("chat/completions", json=payload)
                response.raise_for_status()
                raw_content = response.json()["choices"][0]["message"]["content"]
                if isinstance(raw_content, list):
                    raw_content = "".join(
                        item.get("text", "") for item in raw_content if isinstance(item, dict)
                    )
                return response_model.model_validate_json(self._strip_fences(raw_content))
            except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
        raise ModelGatewayError(f"{role} failed after 3 attempts: {last_error}") from last_error

    @staticmethod
    async def _as_data_url(path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = await asyncio.to_thread(path.read_bytes)
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _strip_fences(value: str) -> str:
        value = value.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines)
        return value
