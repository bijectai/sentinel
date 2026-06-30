from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StartRunRequest(BaseModel):
    policy_id: str
    agent: Literal["scripted", "claude", "gpt"]
    mode: Literal["single", "compare"]
    max_rounds: int = 10
    guardrail_base_url: str
    api_key: str | None = None
    fixtures_dir: str = "examples/policies"


class StartRunResponse(BaseModel):
    run_id: str


class RunStatus(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"]
    mode: Literal["single", "compare"]
    policy_id: str
    agent_name: str
    # For single mode: list of RoundResult dicts
    # For compare mode: dict[guardrail_name, list[RoundResult dicts]]
    rounds: Any
    # Compare mode only
    disagreement_rounds: list[dict[str, Any]] | None = None
    error: str | None = None


class WsMessage(BaseModel):
    type: Literal["round", "round_compare", "complete", "error"]
    data: Any
