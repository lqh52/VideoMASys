from pathlib import Path

from videomasys.domain.schemas import SceneAnalysis
from videomasys.model_gateway.vlm import VLMGateway


class SceneAnalyst:
    def __init__(self, gateway: VLMGateway) -> None:
        self.gateway = gateway

    async def analyze(
        self, frame_paths: list[Path], start_seconds: float, end_seconds: float
    ) -> SceneAnalysis:
        prompt = f"""
Analyze the ordered representative frames from the candidate scene at
{start_seconds:.3f}s–{end_seconds:.3f}s. Write a literal, retrieval-friendly caption
covering people/objects, actions, setting, and visible text. The frames are sparse:
do not claim continuous motion unless supported. boundary_confidence is 0–1 and
measures how likely this scene completes a semantic action or transition. Timestamps
are metadata, not evidence of what happened.
""".strip()
        return await self.gateway.complete_structured(
            role="scene/event analyst",
            prompt=prompt,
            response_model=SceneAnalysis,
            image_paths=frame_paths,
            temperature=0.2,
            max_tokens=500,
        )
