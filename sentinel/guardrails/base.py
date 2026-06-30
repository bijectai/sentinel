from __future__ import annotations

import abc
from typing import Any, Literal

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    outcome: Literal["proved", "refuted", "timeout", "error"]
    reject_code: str | None = None
    explanation: str | None = None
    latency_us: int | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class GuardrailAdapter(abc.ABC):
    @abc.abstractmethod
    async def verify(
        self,
        tool_name: str,
        params: dict[str, Any],
        agent_id: str,
    ) -> Verdict: ...
