from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from sentinel.guardrails.base import GuardrailAdapter, Verdict


class StubRule(BaseModel):
    tool_name: str
    param_key: str
    # Threshold is compared against a derived ratio or direct param value.
    # Units depend on the rule — see _DEFAULT_RULES for documentation per rule.
    max_value: float
    description: str = ""


# Default rules mirror CAP-001 and POS-001 semantics for direct comparability
# with BijectAdapter. Units differ from biject's internal representation:
#
#   place_order:        ratio = params["qty"] / params["available_capital"]
#                       refuted if ratio > 0.10  (biject: CAP-001, 10% threshold)
#                       param_key="qty" is the numerator; available_capital is the
#                       denominator — the rule logic is special-cased in verify().
#
#   rebalance_portfolio: new_weight is a fraction in [0, 1], not basis points.
#                        refuted if new_weight > 0.25  (biject: POS-001, 25% threshold)
#                        param_key="new_weight" is compared directly to max_value.
_DEFAULT_RULES: list[StubRule] = [
    StubRule(
        tool_name="place_order",
        param_key="qty",
        max_value=0.10,
        description="Order qty must not exceed 10% of available_capital (CAP-001 mirror)",
    ),
    StubRule(
        tool_name="rebalance_portfolio",
        param_key="new_weight",
        max_value=0.25,
        description="Portfolio weight must not exceed 25% (POS-001 mirror, fraction units)",
    ),
]


def _build_rule_index(rules: list[StubRule]) -> dict[str, StubRule]:
    return {r.tool_name: r for r in rules}


class StubAdapter(GuardrailAdapter):
    """Fully local, deterministic threshold-check guardrail.

    No network calls — useful as a stable test fixture and as a real second
    guardrail target for sentinel compare.
    """

    def __init__(self, rules: dict[str, StubRule] | None = None) -> None:
        if rules is not None:
            self._rules = rules
        else:
            self._rules = _build_rule_index(_DEFAULT_RULES)

    async def verify(
        self,
        tool_name: str,
        params: dict[str, Any],
        agent_id: str,
    ) -> Verdict:
        t0 = time.monotonic()

        rule = self._rules.get(tool_name)
        if rule is None:
            latency_us = int((time.monotonic() - t0) * 1_000_000)
            return Verdict(
                outcome="error",
                explanation=f"No stub rule for tool_name={tool_name}",
                latency_us=latency_us,
            )

        # Special case: place_order threshold is a ratio of qty / available_capital.
        if tool_name == "place_order":
            qty = params.get("qty")
            capital = params.get("available_capital")
            if qty is None or capital is None:
                latency_us = int((time.monotonic() - t0) * 1_000_000)
                return Verdict(
                    outcome="error",
                    explanation="place_order requires params 'qty' and 'available_capital'",
                    latency_us=latency_us,
                )
            try:
                ratio = float(qty) / float(capital)
            except (TypeError, ZeroDivisionError, ValueError) as e:
                latency_us = int((time.monotonic() - t0) * 1_000_000)
                return Verdict(
                    outcome="error",
                    explanation=f"Cannot compute qty/available_capital ratio: {e}",
                    latency_us=latency_us,
                )
            latency_us = int((time.monotonic() - t0) * 1_000_000)
            if ratio > rule.max_value:
                return Verdict(
                    outcome="refuted",
                    reject_code="STUB_THRESHOLD_BREACH",
                    explanation=(
                        f"qty/available_capital={ratio:.4f} exceeds max {rule.max_value} "
                        f"({rule.description})"
                    ),
                    latency_us=latency_us,
                )
            return Verdict(
                outcome="proved",
                explanation=f"qty/available_capital={ratio:.4f} within limit {rule.max_value}",
                latency_us=latency_us,
            )

        # General case: compare param_key value directly to max_value.
        raw = params.get(rule.param_key)
        if raw is None:
            latency_us = int((time.monotonic() - t0) * 1_000_000)
            return Verdict(
                outcome="error",
                explanation=f"Missing param '{rule.param_key}' for tool '{tool_name}'",
                latency_us=latency_us,
            )
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            latency_us = int((time.monotonic() - t0) * 1_000_000)
            return Verdict(
                outcome="error",
                explanation=f"Non-numeric param '{rule.param_key}': {e}",
                latency_us=latency_us,
            )

        latency_us = int((time.monotonic() - t0) * 1_000_000)
        if value > rule.max_value:
            return Verdict(
                outcome="refuted",
                reject_code="STUB_THRESHOLD_BREACH",
                explanation=(
                    f"{rule.param_key}={value} exceeds max {rule.max_value} "
                    f"({rule.description})"
                ),
                latency_us=latency_us,
            )
        return Verdict(
            outcome="proved",
            explanation=f"{rule.param_key}={value} within limit {rule.max_value}",
            latency_us=latency_us,
        )
