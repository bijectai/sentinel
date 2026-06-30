from __future__ import annotations

import sys
from typing import Any

from pydantic import BaseModel

from sentinel.agents.base import AgentAdapter, AgentContext, ProposedAction
from sentinel.agents.claude_agent import AgentError
from sentinel.agents.scripted_agent import NoMoreActionsError
from sentinel.guardrails.base import GuardrailAdapter
from sentinel.policies.registry import PolicyUnderTest
from sentinel.runner.classifier import classify
from sentinel.runner.loop import (
    RoundResult,
    RunResult,
    _applies_to_tools,
    _match_case_label,
    _synthesize_description,
)


class ComparisonResult(BaseModel):
    policy_id: str
    agent_name: str
    # Keyed by guardrail name (insertion order preserved in Python 3.7+).
    guardrail_results: dict[str, RunResult]
    # Rounds where at least two guardrails disagreed on the same proposed action.
    disagreement_rounds: list[dict[str, Any]]


async def run_comparison(
    agent: AgentAdapter,
    guardrails: dict[str, GuardrailAdapter],
    policy: PolicyUnderTest,
    agent_id: str,
    max_rounds: int = 10,
    agent_name: str = "unknown",
) -> ComparisonResult:
    """Run one agent against multiple guardrails, using a single proposed action per round.

    Design note — shared proposal, divergent verdicts:
        agent.propose_action() is called EXACTLY ONCE per round, then the same
        ProposedAction is dispatched to every guardrail in parallel (sequentially in
        the current implementation, but logically simultaneous). This is the guarantee
        that makes the comparison meaningful: identical input, different guardrail outputs.

    Design note — canonical history for agent context:
        After each round, the agent's history is updated using the verdict from the
        FIRST guardrail in insertion order. This is an acknowledged simplification: the
        agent needs a single coherent narrative to reason against; it cannot be shown
        multiple parallel verdicts. Using the first guardrail as canonical is arbitrary
        but explicit and reproducible. Callers should be aware that agent behavior in
        later rounds is shaped by that guardrail's verdicts, not by all guardrails equally.
    """
    description = _synthesize_description(policy)
    applies_to_tools = _applies_to_tools(policy)
    guardrail_names = list(guardrails.keys())

    # Per-guardrail round accumulators.
    rounds_per_guardrail: dict[str, list[RoundResult]] = {name: [] for name in guardrail_names}
    disagreement_rounds: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    canonical_name = guardrail_names[0]  # first guardrail feeds agent history

    for round_number in range(1, max_rounds + 1):
        context = AgentContext(
            policy_id=policy.policy_id,
            policy_description=description,
            applies_to_tools=applies_to_tools,
            history=list(history),
        )

        # Single propose_action call — never called once-per-guardrail.
        try:
            action: ProposedAction = await agent.propose_action(context)
        except NoMoreActionsError:
            break
        except AgentError as e:
            print(
                f"[AGENT FAILURE] round {round_number} failed at the agent layer "
                f"(no verdict produced, round skipped): {e}",
                file=sys.stderr,
            )
            break

        case_label = _match_case_label(action, policy)
        outcomes_this_round: dict[str, str] = {}
        canonical_verdict = None

        for name, guardrail in guardrails.items():
            verdict = await guardrail.verify(action.tool_name, action.params, agent_id)
            classification = classify(case_label, verdict)

            rounds_per_guardrail[name].append(
                RoundResult(
                    round_number=round_number,
                    proposed_action=action,
                    verdict=verdict,
                    classification=classification,
                    case_label=case_label,
                )
            )
            outcomes_this_round[name] = verdict.outcome

            if name == canonical_name:
                canonical_verdict = verdict

        # Disagreement: any outcome differs from the first guardrail's outcome.
        first_outcome = outcomes_this_round[canonical_name]
        if any(o != first_outcome for o in outcomes_this_round.values()):
            disagreement_rounds.append(
                {
                    "round_number": round_number,
                    "tool_name": action.tool_name,
                    "params": action.params,
                    "results": dict(outcomes_this_round),
                }
            )

        # Feed canonical verdict into agent history.
        assert canonical_verdict is not None
        history.append(
            {
                "tool_name": action.tool_name,
                "params": action.params,
                "verdict": canonical_verdict.model_dump(),
                "classification": classify(case_label, canonical_verdict),
            }
        )

    guardrail_results: dict[str, RunResult] = {}
    for name in guardrail_names:
        rounds = rounds_per_guardrail[name]
        guardrail_results[name] = RunResult(
            policy_id=policy.policy_id,
            agent_name=agent_name,
            guardrail_name=name,
            rounds=rounds,
            bypass_count=sum(1 for r in rounds if r.classification == "BYPASSED"),
            infra_gap_count=sum(1 for r in rounds if r.classification == "INFRA_GAP"),
        )

    return ComparisonResult(
        policy_id=policy.policy_id,
        agent_name=agent_name,
        guardrail_results=guardrail_results,
        disagreement_rounds=disagreement_rounds,
    )
