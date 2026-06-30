from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    policy_id: str
    policy_description: str
    applies_to_tools: list[str]
    # Each entry: {"tool_name": str, "params": dict, "verdict": {"outcome": str, ...}}
    # Append after each round so multi-round loops can see the full attempt history.
    history: list[dict[str, Any]] = Field(default_factory=list)


class ProposedAction(BaseModel):
    tool_name: str
    params: dict[str, Any]
    reasoning: str = ""


class AgentAdapter(abc.ABC):
    @abc.abstractmethod
    async def propose_action(self, context: AgentContext) -> ProposedAction:
        """Given the current policy context and attempt history, propose the next tool call."""
