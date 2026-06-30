from __future__ import annotations

import time
from typing import Any

import httpx

from sentinel.guardrails.base import GuardrailAdapter, Verdict

_PROVED = {"proved", "allowed"}
_REFUTED = {"refuted", "blocked"}


class BijectAdapter(GuardrailAdapter):
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    async def verify(
        self,
        tool_name: str,
        params: dict[str, Any],
        agent_id: str,
    ) -> Verdict:
        t0 = time.monotonic()
        try:
            response = await self._client.post(
                "/api/verify",
                json={"tool_name": tool_name, "params": params, "agent_id": agent_id},
            )
            latency_us = int((time.monotonic() - t0) * 1_000_000)

            try:
                body: dict[str, Any] = response.json()
            except Exception:
                body = {}

            if not response.is_success:
                return Verdict(
                    outcome="error",
                    explanation=f"HTTP {response.status_code}",
                    latency_us=latency_us,
                    raw_response=body,
                )

            raw_verdict = body.get("verdict", "")
            if raw_verdict in _PROVED:
                outcome = "proved"
            elif raw_verdict in _REFUTED:
                outcome = "refuted"
            elif raw_verdict == "timeout":
                outcome = "timeout"
            else:
                outcome = "error"

            return Verdict(
                outcome=outcome,
                reject_code=body.get("reject_code"),
                explanation=body.get("explanation"),
                latency_us=latency_us,
                raw_response=body,
            )

        except httpx.TimeoutException as e:
            latency_us = int((time.monotonic() - t0) * 1_000_000)
            return Verdict(outcome="timeout", explanation=str(e), latency_us=latency_us)
        except Exception as e:
            latency_us = int((time.monotonic() - t0) * 1_000_000)
            return Verdict(outcome="error", explanation=str(e), latency_us=latency_us)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BijectAdapter":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
