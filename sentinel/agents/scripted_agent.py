from __future__ import annotations

from sentinel.agents.base import AgentAdapter, AgentContext, ProposedAction
from sentinel.policies.registry import ToolCallCase


class NoMoreActionsError(Exception):
    """Raised when ScriptedAgent has exhausted its case list."""


class ScriptedAgent(AgentAdapter):
    def __init__(self, cases: list[ToolCallCase]) -> None:
        self._cases = cases
        self._index = 0

    async def propose_action(self, context: AgentContext) -> ProposedAction:
        if self._index >= len(self._cases):
            raise NoMoreActionsError(
                f"ScriptedAgent exhausted all {len(self._cases)} cases"
            )
        case = self._cases[self._index]
        self._index += 1
        return ProposedAction(
            tool_name=case.tool_name,
            params=case.params,
            reasoning="scripted replay",
        )
