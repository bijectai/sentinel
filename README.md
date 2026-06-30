# Sentinel

Sentinel is an open-source adversarial testing tool for AI guardrail systems. It loads a compliance policy, points one or more guardrails at it, and runs an AI agent — scripted or live LLM — against them to see whether the guardrail correctly blocks what it should and allows what it should.

Sentinel ships with [biject](https://bijectai.com) as its reference guardrail implementation, but the agent and guardrail layers are both pluggable.

## What a `BYPASSED` result means

If Sentinel reports `BYPASSED`, it means **the specific policy or guardrail configuration under test let through an action it should have blocked.** It does not mean the underlying guardrail technology is broken — a bypass is a finding about a deployed policy, not an indictment of the product. Treat every `BYPASSED` result as something to investigate against your own policy configuration first.

## Install

```bash
git clone https://github.com/bijectai/sentinel
cd sentinel
pip install -e .
```

Requires Python 3.10+.

## Quickstart

1. Copy `sentinel.yaml` and point it at your guardrail's API and the policy you want to test:

```yaml
target_base_url: "https://api.devrashie.space"
policy_id: "CAP-001"
fixtures_dir: "examples/policies/CAP-001"
agent_id: "sentinel-v0"
```

2. Each policy needs ground-truth fixtures: a directory containing `expected_proved.json` and `expected_refuted.json`, listing tool calls and the verdict you expect for each. See `examples/policies/` for working examples. The `tool_name` and `params` in your fixtures must match what your guardrail's policy actually expects — check your policy's registry/schema, not just the test you're trying to write.

3. Run a basic fixture check:

```bash
sentinel run --config sentinel.yaml
```

This replays each fixture case once and reports PASS/MISMATCH per case.

## Commands

### `sentinel list-policies`

List available policy fixture directories.

```bash
sentinel list-policies --fixtures-dir examples/policies
```

### `sentinel run`

Replay all fixture cases for a policy once, comparing actual guardrail outcome to expected outcome.

```bash
sentinel run --config sentinel.yaml
```

Exit code 0 if all cases match expectation, 1 if any mismatch or error.

### `sentinel adversarial`

Run a live, multi-round adversarial loop against a single guardrail. Each round, the agent proposes a tool call, the guardrail verifies it, and the result is classified and fed back into the agent's next attempt.

```bash
sentinel adversarial --config sentinel.yaml --agent scripted --max-rounds 10
sentinel adversarial --config sentinel.yaml --agent claude --max-rounds 10
sentinel adversarial --config sentinel.yaml --agent gpt --max-rounds 10
```

`--agent scripted` replays your fixture cases in order — useful as a deterministic baseline. `--agent claude` uses a live Claude call each round to actively probe policy boundaries, using the history of previous rounds to inform its next attempt. Requires `ANTHROPIC_API_KEY`. `--agent gpt` does the same using GPT-4o. Requires `OPENAI_API_KEY`.

Exit code 2 if any round is classified `BYPASSED`, 0 otherwise.

### `sentinel compare`

Run the same agent against two guardrails in parallel — biject and a local deterministic `StubAdapter` — using identical proposed actions each round, so verdicts can be compared directly.

```bash
sentinel compare --config sentinel.yaml --agent scripted --max-rounds 10
sentinel compare --config sentinel.yaml --agent claude --max-rounds 10
sentinel compare --config sentinel.yaml --agent gpt --max-rounds 10
```

Surfaces both bypasses and **disagreements** (rounds where the two guardrails returned different verdicts for the same action) as separate findings. A disagreement alone is not a failure — it means the two guardrails enforce the policy differently, which is worth investigating but isn't necessarily a security problem.

Exit code 2 if either guardrail produces a `BYPASSED` result, 0 otherwise (disagreements do not affect exit code).

### `sentinel deploy`

Compile and deploy a Lean policy to the configured biject endpoint.

```bash
sentinel deploy --config sentinel.yaml --lean-file path/to/policy.lean --policy-id CUSTOM-001 --description "My policy"
```

Reads `target_base_url` and `api_key` from your config, posts the Lean source to the biject compile endpoint, and reports success or the compile error. Exit code 0 on success, 1 on failure.

## Classification labels

Every round is classified as exactly one of:

| Label | Meaning |
|---|---|
| `BYPASSED` | A known-bad action was incorrectly allowed. The critical failure case. |
| `BLOCKED_CORRECTLY` | A known-bad action was correctly refused. |
| `ALLOWED_CORRECTLY` | A known-good action was correctly allowed. |
| `INDETERMINATE` | A known-good action was incorrectly blocked (over-strict guardrail — a real finding, not a bypass). |
| `INFRA_GAP` | The guardrail errored or timed out and produced no real verdict — no correctness claim can be made. |

When an agent explores freely with no matching fixture case (autonomous mode with no declared ground truth), only `BLOCKED_CORRECTLY` / `ALLOWED_CORRECTLY` are used, and they describe what the guardrail *did*, not a verified-correct outcome — there's no ground truth to check it against.

## Writing your own guardrail adapter

Implement `GuardrailAdapter` (`sentinel/guardrails/base.py`):

```python
class GuardrailAdapter(abc.ABC):
    async def verify(self, tool_name: str, params: dict, agent_id: str) -> Verdict:
        ...
```

`verify()` must never raise — all failure modes should resolve to `Verdict(outcome="error", ...)` or `Verdict(outcome="timeout", ...)`.

## Writing your own agent adapter

Implement `AgentAdapter` (`sentinel/agents/base.py`):

```python
class AgentAdapter(abc.ABC):
    async def propose_action(self, context: AgentContext) -> ProposedAction:
        ...
```

`AgentContext` includes the policy under test and the history of previous rounds and their verdicts, so an agent can adapt its strategy round over round.

## Status

Sentinel is under active development. Current scope: scripted, Claude, and GPT-4o agents; biject and stub guardrail adapters; single-guardrail adversarial mode; two-guardrail comparison mode; and Lean policy deployment via `sentinel deploy`. The GUI (`web/index.html`) is a self-contained simulation for demos and does not require a running backend.

## License

MIT
