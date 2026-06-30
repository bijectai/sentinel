from __future__ import annotations

import asyncio
import os
from typing import Any

import typer

from sentinel.agents.base import AgentAdapter
from sentinel.agents.claude_agent import AgentError, ClaudeAgent
from sentinel.agents.scripted_agent import ScriptedAgent
from sentinel.config import SentinelConfigError, load_config
from sentinel.guardrails.base import GuardrailAdapter, Verdict
from sentinel.guardrails.biject_adapter import BijectAdapter
from sentinel.policies.registry import PolicyUnderTest, load_policy
from sentinel.guardrails.stub_adapter import StubAdapter
from sentinel.runner.comparator import ComparisonResult, run_comparison
from sentinel.runner.loop import RoundResult, RunResult, run_loop

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config_path: str = typer.Option("sentinel.yaml", "--config", help="Path to sentinel.yaml"),
) -> None:
    """Run all policy fixture cases against the configured guardrail endpoint."""
    try:
        cfg = load_config(config_path)
    except SentinelConfigError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(1)

    try:
        policy = load_policy(cfg.policy_id, cfg.fixtures_dir)
    except SentinelConfigError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(1)

    adapter: GuardrailAdapter = BijectAdapter(cfg.target_base_url, cfg.api_key)

    matched, total = asyncio.run(_run_cases(adapter, policy, cfg.agent_id))

    typer.echo(f"\n{matched}/{total} cases matched expected outcome.")
    raise typer.Exit(0 if matched == total else 1)


async def _run_cases(
    adapter: GuardrailAdapter,
    policy: PolicyUnderTest,
    agent_id: str,
) -> tuple[int, int]:
    cases = [
        (case, "proved") for case in policy.expected_proved
    ] + [
        (case, "refuted") for case in policy.expected_refuted
    ]

    matched = 0
    try:
        for case, expected in cases:
            verdict: Verdict = await adapter.verify(case.tool_name, case.params, agent_id)
            ok = verdict.outcome == expected
            if ok:
                matched += 1
            tag = "[PASS]" if ok else "[MISMATCH]"
            reject = f" reject_code={verdict.reject_code}" if verdict.reject_code else ""
            typer.echo(
                f"{tag} {case.tool_name} {case.params}"
                f" | expected={expected} actual={verdict.outcome}{reject}"
            )
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            await close()

    return matched, len(cases)


@app.command()
def adversarial(
    config_path: str = typer.Option("sentinel.yaml", "--config", help="Path to sentinel.yaml"),
    agent: str = typer.Option("scripted", "--agent", help="scripted or claude"),
    max_rounds: int = typer.Option(10, "--max-rounds"),
) -> None:
    """Run an adversarial agent-vs-guardrail loop and classify each round."""
    try:
        cfg = load_config(config_path)
    except SentinelConfigError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(1)

    try:
        policy = load_policy(cfg.policy_id, cfg.fixtures_dir)
    except SentinelConfigError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(1)

    guardrail: GuardrailAdapter = BijectAdapter(cfg.target_base_url, cfg.api_key)
    guardrail_name = type(guardrail).__name__

    agent_adapter: AgentAdapter
    if agent == "scripted":
        agent_adapter = ScriptedAgent(policy.expected_proved + policy.expected_refuted)
    elif agent == "claude":
        try:
            agent_adapter = ClaudeAgent()
        except AgentError as e:
            typer.echo(f"[ERROR] {e}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"[ERROR] Unknown agent '{agent}' (expected 'scripted' or 'claude')", err=True)
        raise typer.Exit(1)

    result = asyncio.run(
        _run_adversarial(
            agent_adapter, guardrail, policy, cfg.agent_id, max_rounds, agent, guardrail_name
        )
    )

    typer.echo("")
    typer.echo(f"Total rounds: {len(result.rounds)}")
    typer.echo(f"Bypass count: {result.bypass_count}")
    typer.echo(f"Infra-gap count: {result.infra_gap_count}")
    if result.bypass_count > 0:
        typer.secho(
            f"\n⚠ {result.bypass_count} BYPASS(ES) DETECTED — "
            "guardrail allowed an action that should have been refused.",
            fg=typer.colors.RED,
            bold=True,
            err=True,
        )
        raise typer.Exit(2)

    raise typer.Exit(0)


async def _run_adversarial(
    agent_adapter: AgentAdapter,
    guardrail: GuardrailAdapter,
    policy: PolicyUnderTest,
    agent_id: str,
    max_rounds: int,
    agent_name: str,
    guardrail_name: str,
) -> RunResult:
    try:
        result = await run_loop(
            agent=agent_adapter,
            guardrail=guardrail,
            policy=policy,
            agent_id=agent_id,
            max_rounds=max_rounds,
            agent_name=agent_name,
            guardrail_name=guardrail_name,
        )
    finally:
        close = getattr(guardrail, "close", None)
        if close is not None:
            await close()

    # run_loop buffers rounds internally; print them now so the operator sees a
    # per-round trace. (Live streaming arrives in a later session.)
    for r in result.rounds:
        _print_round(r)

    return result


def _print_round(r: RoundResult) -> None:
    label = r.case_label if r.case_label is not None else "none"
    base = (
        f"Round {r.round_number}: {r.proposed_action.tool_name} "
        f"{r.proposed_action.params} | verdict={r.verdict.outcome} "
        f"case_label={label} -> {r.classification}"
    )
    if r.classification == "BYPASSED":
        typer.secho("=" * 72, fg=typer.colors.RED, bold=True)
        typer.secho(f">>> BYPASSED <<<  {base}", fg=typer.colors.RED, bold=True)
        typer.secho("=" * 72, fg=typer.colors.RED, bold=True)
    else:
        typer.echo(base)


@app.command()
def compare(
    config_path: str = typer.Option("sentinel.yaml", "--config", help="Path to sentinel.yaml"),
    agent: str = typer.Option("scripted", "--agent", help="scripted or claude"),
    max_rounds: int = typer.Option(10, "--max-rounds"),
) -> None:
    """Run an agent against biject and stub guardrails side-by-side and report disagreements."""
    try:
        cfg = load_config(config_path)
    except SentinelConfigError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(1)

    try:
        policy = load_policy(cfg.policy_id, cfg.fixtures_dir)
    except SentinelConfigError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(1)

    guardrails: dict[str, Any] = {
        "biject": BijectAdapter(cfg.target_base_url, cfg.api_key),
        "stub": StubAdapter(),
    }

    agent_adapter: AgentAdapter
    if agent == "scripted":
        agent_adapter = ScriptedAgent(policy.expected_proved + policy.expected_refuted)
    elif agent == "claude":
        try:
            agent_adapter = ClaudeAgent()
        except AgentError as e:
            typer.echo(f"[ERROR] {e}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"[ERROR] Unknown agent '{agent}' (expected 'scripted' or 'claude')", err=True)
        raise typer.Exit(1)

    result = asyncio.run(
        _run_comparison(agent_adapter, guardrails, policy, cfg.agent_id, max_rounds, agent)
    )

    _print_comparison(result)

    any_bypass = any(r.bypass_count > 0 for r in result.guardrail_results.values())
    raise typer.Exit(2 if any_bypass else 0)


async def _run_comparison(
    agent_adapter: AgentAdapter,
    guardrails: dict[str, Any],
    policy: PolicyUnderTest,
    agent_id: str,
    max_rounds: int,
    agent_name: str,
) -> ComparisonResult:
    try:
        result = await run_comparison(
            agent=agent_adapter,
            guardrails=guardrails,
            policy=policy,
            agent_id=agent_id,
            max_rounds=max_rounds,
            agent_name=agent_name,
        )
    finally:
        for g in guardrails.values():
            close = getattr(g, "close", None)
            if close is not None:
                await close()
    return result


def _print_comparison(result: ComparisonResult) -> None:
    guardrail_names = list(result.guardrail_results.keys())

    # Collect all rounds from canonical (first) guardrail to iterate.
    canonical = guardrail_names[0]
    canonical_rounds = result.guardrail_results[canonical].rounds
    if not canonical_rounds:
        typer.echo("No rounds completed.")
        return

    # Per-round table header.
    col_width = 16
    header = f"{'Round':<6}  {'Tool / Params':<36}" + "".join(
        f"  {name:<{col_width}}" for name in guardrail_names
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    # Build index of rounds per guardrail for easy lookup.
    rounds_by_guardrail: dict[str, dict[int, RoundResult]] = {
        name: {r.round_number: r for r in result.guardrail_results[name].rounds}
        for name in guardrail_names
    }

    disagreement_set = {d["round_number"] for d in result.disagreement_rounds}

    for ref_round in canonical_rounds:
        rn = ref_round.round_number
        action_str = f"{ref_round.proposed_action.tool_name} {ref_round.proposed_action.params}"
        if len(action_str) > 36:
            action_str = action_str[:33] + "..."
        row = f"{rn:<6}  {action_str:<36}"
        for name in guardrail_names:
            r = rounds_by_guardrail[name].get(rn)
            if r is None:
                cell = f"{'—':<{col_width}}"
            else:
                cell = f"{r.verdict.outcome}/{r.classification}"[:col_width]
                cell = f"{cell:<{col_width}}"
            row += f"  {cell}"

        if rn in disagreement_set:
            typer.secho(row, fg=typer.colors.YELLOW)
        else:
            typer.echo(row)

    # Disagreement block.
    if result.disagreement_rounds:
        typer.echo("")
        typer.secho("⚠  DISAGREEMENTS", fg=typer.colors.YELLOW, bold=True)
        for d in result.disagreement_rounds:
            outcomes = ", ".join(f"{k}={v}" for k, v in d["results"].items())
            typer.secho(
                f"  Round {d['round_number']}: {d['tool_name']} {d['params']} — {outcomes}",
                fg=typer.colors.YELLOW,
            )

    # Per-guardrail summary.
    typer.echo("")
    typer.echo("Summary:")
    for name, run in result.guardrail_results.items():
        typer.echo(
            f"  {name}: rounds={len(run.rounds)}  bypass={run.bypass_count}"
            f"  infra_gap={run.infra_gap_count}"
        )
    typer.echo(f"  disagreements: {len(result.disagreement_rounds)}")

    # Bypass banners.
    for name, run in result.guardrail_results.items():
        if run.bypass_count > 0:
            typer.secho(
                f"\n⚠ {run.bypass_count} BYPASS(ES) on '{name}' — "
                "guardrail allowed an action that should have been refused.",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )


@app.command("list-policies")
def list_policies(
    fixtures_dir: str = typer.Option("examples/policies", "--fixtures-dir"),
) -> None:
    """List available policy IDs by scanning the fixtures directory."""
    try:
        entries = [e.name for e in os.scandir(fixtures_dir) if e.is_dir()]
    except FileNotFoundError:
        typer.echo(f"[ERROR] Fixtures directory not found: {fixtures_dir}", err=True)
        raise typer.Exit(1)

    if not entries:
        typer.echo("No policies found.")
        return

    for name in sorted(entries):
        typer.echo(name)
