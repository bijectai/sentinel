from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel


class DeployResult(BaseModel):
    success: bool
    policy_id: str
    error: str | None = None
    raw_response: dict[str, Any] = {}


async def deploy_policy(
    base_url: str,
    lean_code: str,
    policy_id: str,
    description: str = "",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> DeployResult:
    """Compile and deploy a Lean policy to the biject endpoint. Never raises."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "policy_id": policy_id,
        "lean_code": lean_code,
        "description": description,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in ("/api/policies/compile", "/api/compile"):
            url = base_url.rstrip("/") + path
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as e:
                return DeployResult(
                    success=False,
                    policy_id=policy_id,
                    error=f"HTTP error on {url}: {e}",
                )

            if resp.status_code == 404:
                continue  # try fallback path

            try:
                body: dict[str, Any] = resp.json()
            except Exception:
                body = {"raw": resp.text}

            if resp.is_success:
                return DeployResult(
                    success=True,
                    policy_id=policy_id,
                    raw_response=body,
                )

            return DeployResult(
                success=False,
                policy_id=policy_id,
                error=body.get("error") or body.get("message") or resp.text,
                raw_response=body,
            )

        return DeployResult(
            success=False,
            policy_id=policy_id,
            error=f"Compile endpoint not found at {base_url} (tried /api/policies/compile and /api/compile)",
        )
