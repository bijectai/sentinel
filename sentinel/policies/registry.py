from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from sentinel.config import SentinelConfigError


class ToolCallCase(BaseModel):
    tool_name: str
    params: dict[str, Any]
    description: str = ""


class PolicyUnderTest(BaseModel):
    policy_id: str
    expected_proved: list[ToolCallCase]
    expected_refuted: list[ToolCallCase]


_case_adapter: TypeAdapter[list[ToolCallCase]] = TypeAdapter(list[ToolCallCase])


def load_policy(policy_id: str, fixtures_dir: str) -> PolicyUnderTest:
    # KNOWN GAP: nothing here validates that a fixture's tool_name matches the
    # target registry's applies_to_tools, or that its params align with the
    # policy's parameter_map. A CAP-001 fixture sending "rebalance_portfolio"
    # will load without error but produce meaningless verdicts against the live
    # guardrail. Cross-policy mismatches are a footgun; add structural validation
    # here in a future session once the live registry is queryable at load time.
    proved = _load_fixture(f"{fixtures_dir}/expected_proved.json")
    refuted = _load_fixture(f"{fixtures_dir}/expected_refuted.json")
    return PolicyUnderTest(
        policy_id=policy_id,
        expected_proved=proved,
        expected_refuted=refuted,
    )


def _load_fixture(path: str) -> list[ToolCallCase]:
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise SentinelConfigError(f"Missing fixture file: {path}")
    except json.JSONDecodeError as e:
        raise SentinelConfigError(f"Malformed fixture {path}: {e}")

    try:
        return _case_adapter.validate_python(raw)
    except ValidationError as e:
        raise SentinelConfigError(f"Malformed fixture {path}: {e}")
