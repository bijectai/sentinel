from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sentinel.agents.base import AgentAdapter
from sentinel.agents.claude_agent import AgentError, ClaudeAgent
from sentinel.agents.gpt_agent import GPTAgent
from sentinel.agents.scripted_agent import ScriptedAgent
from sentinel.config import SentinelConfigError
from sentinel.guardrails.base import GuardrailAdapter
from sentinel.guardrails.biject_adapter import BijectAdapter
from sentinel.guardrails.stub_adapter import StubAdapter
from sentinel.policies.registry import PolicyUnderTest, load_policy
from sentinel.runner.comparator import run_comparison
from sentinel.runner.loop import RoundResult, run_loop
from sentinel.web.schemas import RunStatus, StartRunRequest, StartRunResponse, WsMessage

log = logging.getLogger(__name__)

app = FastAPI(title="Sentinel")

_allowed_origins_env = os.environ.get("SENTINEL_CORS_ORIGINS", "")
_allowed_origins: list[str] = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "null",  # file:// origin for browser-opened local HTML
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# In-memory run registry: run_id -> RunState dict
_runs: dict[str, dict[str, Any]] = {}
# Per-run WebSocket subscribers: run_id -> list of active send queues
_ws_queues: dict[str, list[asyncio.Queue]] = {}


# ---------------------------------------------------------------------------
# Static / HTML
# ---------------------------------------------------------------------------

_web_dir = os.path.join(os.path.dirname(__file__))
_public_dir = os.path.join(_web_dir, "..", "..", "web", "public")

if os.path.isdir(_public_dir):
    app.mount("/public", StaticFiles(directory=_public_dir), name="public")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(_web_dir, "index.html"))


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


@app.get("/api/policies")
async def list_policies(fixtures_dir: str = "examples/policies") -> list[str]:
    try:
        entries = [e.name for e in os.scandir(fixtures_dir) if e.is_dir()]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Fixtures directory not found: {fixtures_dir}")
    return sorted(entries)


@app.post("/api/runs", response_model=StartRunResponse)
async def start_run(req: StartRunRequest) -> StartRunResponse:
    try:
        policy = load_policy(req.policy_id, req.fixtures_dir + "/" + req.policy_id)
    except SentinelConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build agent — api_key never stored on the run record
    try:
        agent_adapter = _build_agent(req.agent, policy)
    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "run_id": run_id,
        "status": "running",
        "mode": req.mode,
        "policy_id": req.policy_id,
        "agent_name": req.agent,
        "rounds": [] if req.mode == "single" else {},
        "disagreement_rounds": [] if req.mode == "compare" else None,
        "error": None,
    }
    _ws_queues[run_id] = []

    asyncio.create_task(
        _execute_run(run_id, req, agent_adapter, policy)
    )

    return StartRunResponse(run_id=run_id)


@app.get("/api/runs/{run_id}", response_model=RunStatus)
async def get_run(run_id: str) -> RunStatus:
    state = _runs.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatus(**state)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()

    state = _runs.get(run_id)
    if state is None:
        await websocket.send_json({"type": "error", "data": "Run not found"})
        await websocket.close()
        return

    queue: asyncio.Queue = asyncio.Queue()
    _ws_queues[run_id].append(queue)

    try:
        # Replay accumulated rounds so a late-connecting client misses nothing.
        await _replay_rounds(websocket, state)

        # If already done, send complete and close.
        if state["status"] != "running":
            await websocket.send_json(
                WsMessage(type="complete", data={"status": state["status"], "error": state["error"]}).model_dump()
            )
            return

        # Live tail: pull from queue until run ends or client disconnects.
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive
                await websocket.send_json({"type": "ping"})
                continue

            await websocket.send_json(msg)
            if msg.get("type") == "complete":
                break

    except WebSocketDisconnect:
        pass
    finally:
        queues = _ws_queues.get(run_id, [])
        if queue in queues:
            queues.remove(queue)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _replay_rounds(websocket: WebSocket, state: dict) -> None:
    """Send all rounds accumulated so far to a newly-connected WebSocket."""
    if state["mode"] == "single":
        for round_data in state["rounds"]:
            await websocket.send_json(
                WsMessage(type="round", data=round_data).model_dump()
            )
    else:
        # compare mode: rounds is dict[guardrail_name, list[round_data]]
        # Reconstruct per-round messages keyed by round_number
        rounds_by_number: dict[int, dict[str, Any]] = {}
        for guardrail_name, round_list in state["rounds"].items():
            for rd in round_list:
                rn = rd["round_number"]
                if rn not in rounds_by_number:
                    rounds_by_number[rn] = {"round_number": rn, "results": {}}
                rounds_by_number[rn]["results"][guardrail_name] = rd

        for rn in sorted(rounds_by_number.keys()):
            await websocket.send_json(
                WsMessage(type="round_compare", data=rounds_by_number[rn]).model_dump()
            )


async def _broadcast(run_id: str, msg: dict) -> None:
    for queue in list(_ws_queues.get(run_id, [])):
        await queue.put(msg)


def _build_agent(agent_name: str, policy: PolicyUnderTest) -> AgentAdapter:
    if agent_name == "scripted":
        return ScriptedAgent(policy.expected_proved + policy.expected_refuted)
    if agent_name == "claude":
        return ClaudeAgent()
    if agent_name == "gpt":
        return GPTAgent()
    raise ValueError(f"Unknown agent '{agent_name}'")


def _round_to_dict(r: RoundResult) -> dict:
    return r.model_dump()


async def _execute_run(
    run_id: str,
    req: StartRunRequest,
    agent_adapter: AgentAdapter,
    policy: PolicyUnderTest,
) -> None:
    # api_key is used only here to construct the adapter; never stored on state.
    guardrail: GuardrailAdapter = BijectAdapter(req.guardrail_base_url, req.api_key)

    try:
        if req.mode == "single":
            await _run_single(run_id, req, agent_adapter, guardrail, policy)
        else:
            await _run_compare(run_id, req, agent_adapter, guardrail, policy)
    except Exception as exc:
        log.exception("Unexpected error in background run %s", run_id)
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["error"] = str(exc)
        await _broadcast(
            run_id,
            WsMessage(type="error", data=str(exc)).model_dump(),
        )
        await _broadcast(
            run_id,
            WsMessage(type="complete", data={"status": "failed", "error": str(exc)}).model_dump(),
        )
    finally:
        close = getattr(guardrail, "close", None)
        if close is not None:
            await close()


async def _run_single(
    run_id: str,
    req: StartRunRequest,
    agent_adapter: AgentAdapter,
    guardrail: GuardrailAdapter,
    policy: PolicyUnderTest,
) -> None:
    # Monkey-patch: intercept rounds as they complete by wrapping run_loop.
    # run_loop accumulates internally and returns; we replicate its loop here
    # so we can push each round over the WebSocket immediately.
    from sentinel.agents.claude_agent import AgentError
    from sentinel.agents.scripted_agent import NoMoreActionsError
    from sentinel.runner.classifier import classify
    from sentinel.runner.loop import (
        _applies_to_tools,
        _match_case_label,
        _synthesize_description,
    )

    description = _synthesize_description(policy)
    applies_to_tools = _applies_to_tools(policy)
    from sentinel.agents.base import AgentContext

    history: list[dict] = []
    state = _runs[run_id]

    for round_number in range(1, req.max_rounds + 1):
        context = AgentContext(
            policy_id=policy.policy_id,
            policy_description=description,
            applies_to_tools=applies_to_tools,
            history=list(history),
        )
        try:
            action = await agent_adapter.propose_action(context)
        except NoMoreActionsError:
            break
        except AgentError as e:
            log.warning("[AGENT FAILURE] run=%s round=%d: %s", run_id, round_number, e)
            break

        case_label = _match_case_label(action, policy)
        verdict = await guardrail.verify(action.tool_name, action.params, "sentinel")
        classification = classify(case_label, verdict)

        round_result = RoundResult(
            round_number=round_number,
            proposed_action=action,
            verdict=verdict,
            classification=classification,
            case_label=case_label,
        )
        rd = _round_to_dict(round_result)
        state["rounds"].append(rd)

        await _broadcast(run_id, WsMessage(type="round", data=rd).model_dump())

        history.append(
            {
                "tool_name": action.tool_name,
                "params": action.params,
                "verdict": verdict.model_dump(),
                "classification": classification,
            }
        )

    state["status"] = "completed"
    await _broadcast(
        run_id,
        WsMessage(type="complete", data={"status": "completed", "error": None}).model_dump(),
    )


async def _run_compare(
    run_id: str,
    req: StartRunRequest,
    agent_adapter: AgentAdapter,
    guardrail: GuardrailAdapter,
    policy: PolicyUnderTest,
) -> None:
    from sentinel.agents.base import AgentContext
    from sentinel.agents.claude_agent import AgentError
    from sentinel.agents.scripted_agent import NoMoreActionsError
    from sentinel.runner.classifier import classify
    from sentinel.runner.loop import (
        _applies_to_tools,
        _match_case_label,
        _synthesize_description,
    )

    stub = StubAdapter()
    guardrails: dict[str, GuardrailAdapter] = {
        "biject": guardrail,
        "stub": stub,
    }

    description = _synthesize_description(policy)
    applies_to_tools = _applies_to_tools(policy)
    guardrail_names = list(guardrails.keys())
    canonical_name = guardrail_names[0]

    state = _runs[run_id]
    for name in guardrail_names:
        state["rounds"][name] = []

    history: list[dict] = []

    for round_number in range(1, req.max_rounds + 1):
        context = AgentContext(
            policy_id=policy.policy_id,
            policy_description=description,
            applies_to_tools=applies_to_tools,
            history=list(history),
        )
        try:
            action = await agent_adapter.propose_action(context)
        except NoMoreActionsError:
            break
        except AgentError as e:
            log.warning("[AGENT FAILURE] run=%s round=%d: %s", run_id, round_number, e)
            break

        case_label = _match_case_label(action, policy)
        outcomes_this_round: dict[str, str] = {}
        canonical_verdict = None
        round_results_per_guardrail: dict[str, dict] = {}

        for name, gr in guardrails.items():
            verdict = await gr.verify(action.tool_name, action.params, "sentinel")
            classification = classify(case_label, verdict)
            rr = RoundResult(
                round_number=round_number,
                proposed_action=action,
                verdict=verdict,
                classification=classification,
                case_label=case_label,
            )
            rd = _round_to_dict(rr)
            state["rounds"][name].append(rd)
            round_results_per_guardrail[name] = rd
            outcomes_this_round[name] = verdict.outcome
            if name == canonical_name:
                canonical_verdict = verdict

        first_outcome = outcomes_this_round[canonical_name]
        is_disagreement = any(o != first_outcome for o in outcomes_this_round.values())
        if is_disagreement:
            disagreement = {
                "round_number": round_number,
                "tool_name": action.tool_name,
                "params": action.params,
                "results": dict(outcomes_this_round),
            }
            state["disagreement_rounds"].append(disagreement)

        compare_msg = {
            "round_number": round_number,
            "results": round_results_per_guardrail,
            "is_disagreement": is_disagreement,
        }
        await _broadcast(run_id, WsMessage(type="round_compare", data=compare_msg).model_dump())

        assert canonical_verdict is not None
        history.append(
            {
                "tool_name": action.tool_name,
                "params": action.params,
                "verdict": canonical_verdict.model_dump(),
                "classification": classify(case_label, canonical_verdict),
            }
        )

    state["status"] = "completed"
    await _broadcast(
        run_id,
        WsMessage(type="complete", data={"status": "completed", "error": None}).model_dump(),
    )
