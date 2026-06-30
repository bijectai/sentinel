from __future__ import annotations

import sys
from typing import Literal

from pydantic import BaseModel

from sentinel.agents.base import AgentAdapter, AgentContext, ProposedAction
from sentinel.agents.claude_agent import AgentError
from sentinel.agents.scripted_agent import NoMoreActionsError
from sentinel.guardrails.base import GuardrailAdapter, Verdict
from sentinel.policies.registry import PolicyUnderTest
from sentinel.runner.classifier import Classification, classify


class RoundResult(BaseModel):
    round_number: int
    proposed_action: ProposedAction
    verdict: Verdict
    classification: Classification
    case_label: Literal["proved", "refuted"] | None = None


class RunResult(BaseModel):
    policy_id: str
    agent_name: str
    guardrail_name: str
    rounds: list[RoundResult]
    bypass_count: int
    infra_gap_count: int


def _synthesize_description(policy: PolicyUnderTest) -> str:
    """Build a human-readable policy description from fixture case descriptions,
    since PolicyUnderTest has no dedicated description field."""
    proved = [c.description for c in policy.expected_proved if c.description]
    refuted = [c.description for c in policy.expected_refuted if c.description]
    parts = [f"Policy {policy.policy_id}."]
    if proved:
        parts.append("Actions that should be ALLOWED: " + "; ".join(proved) + ".")
    if refuted:
        parts.append("Actions that should be BLOCKED: " + "; ".join(refuted) + ".")
    return " ".join(parts)


def _applies_to_tools(policy: PolicyUnderTest) -> list[str]:
    seen: dict[str, None] = {}
    for case in policy.expected_proved + policy.expected_refuted:
        seen.setdefault(case.tool_name, None)
    return list(seen.keys())


def _match_case_label(
    action: ProposedAction,
    policy: PolicyUnderTest,
) -> Literal["proved", "refuted"] | None:
    """Return the ground-truth label if this action exactly matches a fixture
    case (tool_name + params), else None (free-form exploration)."""
    for case in policy.expected_proved:
        if case.tool_name == action.tool_name and case.params == action.params:
            return "proved"
    for case in policy.expected_refuted:
        if case.tool_name == action.tool_name and case.params == action.params:
            return "refuted"
    return None


async def run_loop(
    agent: AgentAdapter,
    guardrail: GuardrailAdapter,
    policy: PolicyUnderTest,
    agent_id: str,
    max_rounds: int = 10,
    agent_name: str = "unknown",
    guardrail_name: str = "unknown",
) -> RunResult:
    description = _synthesize_description(policy)
    applies_to_tools = _applies_to_tools(policy)

    # History accumulates across rounds; we rebuild AgentContext each iteration
    # from this list rather than mutating a single shared instance.
    history: list[dict] = []
    rounds: list[RoundResult] = []

    for round_number in range(1, max_rounds + 1):
        context = AgentContext(
            policy_id=policy.policy_id,
            policy_description=description,
            applies_to_tools=applies_to_tools,
            history=list(history),
        )

        try:
            action = await agent.propose_action(context)
        except NoMoreActionsError:
            # Scripted agent ran out of fixtures — not an error, just truncate.
            break
        except AgentError as e:
            # Agent-layer failure: never conflate with INFRA_GAP (guardrail-only),
            # never fabricate a verdict. Stop and return completed rounds.
            print(
                f"[AGENT FAILURE] round {round_number} failed at the agent layer "
                f"(no verdict produced, round skipped): {e}",
                file=sys.stderr,
            )
            break

        case_label = _match_case_label(action, policy)

        verdict: Verdict = await guardrail.verify(
            action.tool_name, action.params, agent_id
        )

        classification = classify(case_label, verdict)

        rounds.append(
            RoundResult(
                round_number=round_number,
                proposed_action=action,
                verdict=verdict,
                classification=classification,
                case_label=case_label,
            )
        )

        history.append(
            {
                "tool_name": action.tool_name,
                "params": action.params,
                "verdict": verdict.model_dump(),
                "classification": classification,
            }
        )

    bypass_count = sum(1 for r in rounds if r.classification == "BYPASSED")
    infra_gap_count = sum(1 for r in rounds if r.classification == "INFRA_GAP")

    return RunResult(
        policy_id=policy.policy_id,
        agent_name=agent_name,
        guardrail_name=guardrail_name,
        rounds=rounds,
        bypass_count=bypass_count,
        infra_gap_count=infra_gap_count,
    )
