from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.ast_reviewer import ast_reviewer_node
from agents.coder import coder_node
from agents.execution import execution_node
from agents.planner import planner_node
from agents.vision_critic import vision_critic_node
from state import AgentState, MAX_RETRIES

GenerateRoute = Literal["to_coder", "to_execution", "to_vision", "finish"]


def _current_retry_count(state: AgentState) -> int:
    """Read retry count defensively."""

    try:
        return int(state.get("retry_count", 0))
    except Exception:
        return MAX_RETRIES


def _should_force_end(state: AgentState) -> bool:
    """Hard-stop guard to prevent infinite retry loops."""

    return _current_retry_count(state) >= MAX_RETRIES


def _is_vision_passed(state: AgentState) -> bool:
    """Vision stage passes when structured verdict accepts the output."""

    verdict = str(state.get("vision_verdict", "") or "").strip().upper()
    severity = str(state.get("vision_severity", "") or "").strip().lower()
    vision_error = state.get("vision_error")

    if verdict == "OK":
        return True
    if vision_error is None and verdict in {"", "OK"}:
        return True
    if verdict == "REVISE" and severity == "low":
        return True
    return False


def _is_minor_vision_issue(state: AgentState) -> bool:
    """Detect low-severity vision feedback using structured state first."""

    severity = state.get("vision_severity")
    if isinstance(severity, str) and severity.strip().lower() == "low":
        return True

    vision_error = state.get("vision_error")
    if vision_error is None:
        return False
    text = str(vision_error).strip().lower()
    if not text:
        return False
    minor_tokens = ["\"severity\": \"low\"", "severity: low", "minor", "轻微", "low severity"]
    return any(token in text for token in minor_tokens)


def _is_infra_failure(state: AgentState, stage: str) -> bool:
    """Check whether current failure was classified as infrastructure-related."""

    return state.get("failure_stage") == stage and state.get("failure_type") == "infra"


def _success_patch(reason: str) -> dict[str, object]:
    return {
        "is_success": 1,
        "final_verdict": "success",
        "success_reason": reason,
        "failure_reason": None,
        "failure_stage": None,
        "failure_type": None,
    }


def _failure_patch(*, verdict: str, stage: str, failure_type: str, reason: str) -> dict[str, object]:
    return {
        "is_success": 0,
        "final_verdict": verdict,
        "success_reason": None,
        "failure_reason": reason,
        "failure_stage": stage,
        "failure_type": failure_type,
    }


def route_after_ast(state: AgentState) -> GenerateRoute:
    """Route after AST review."""

    ast_error = str(state.get("ast_error") or "").strip()
    if ast_error:
        if ast_error == "Generated code is empty.":
            return "finish"
        if _should_force_end(state):
            return "finish"
        return "to_coder"
    return "to_execution"


def route_after_execution(state: AgentState) -> GenerateRoute:
    """Route after execution sandbox."""

    if state.get("render_error"):
        return "finish"
    return "to_vision"


def route_after_vision(state: AgentState) -> GenerateRoute:
    """Route after vision critic."""

    if _is_vision_passed(state):
        return "finish"

    if _is_infra_failure(state, "vision"):
        return "finish"

    retry_count = _current_retry_count(state)
    if retry_count >= 3 and _is_minor_vision_issue(state):
        return "finish"

    severity = str(state.get("vision_severity", "") or "").strip().lower()
    issue_count = int(state.get("vision_issue_count", 0) or 0)
    storyboard_present = bool(state.get("storyboard_present"))

    if severity == "medium":
        if storyboard_present and retry_count >= 3 and issue_count <= 4:
            return "finish"
        if retry_count >= 4 and issue_count <= 2:
            return "finish"

    if _should_force_end(state):
        return "finish"
    return "to_coder"


def verdict_node(state: AgentState) -> dict[str, Any]:
    """Terminal node: compute final_verdict from accumulated state fields."""

    if _is_vision_passed(state):
        severity = str(state.get("vision_severity", "") or "").strip().lower()
        if severity == "low":
            return _success_patch("Accepted low-severity visual issue.")
        return _success_patch("Vision review passed.")

    if state.get("failure_type") == "infra":
        stage = state.get("failure_stage") or "unknown"
        reason = str(
            state.get("failure_reason")
            or state.get("render_error")
            or state.get("vision_error")
            or "Infrastructure failure."
        )
        return _failure_patch(verdict="infra_failure", stage=stage, failure_type="infra", reason=reason)

    retry_count = _current_retry_count(state)
    if retry_count >= 3 and _is_minor_vision_issue(state):
        return _success_patch("Accepted low-severity visual issue after retries.")

    severity = str(state.get("vision_severity", "") or "").strip().lower()
    issue_count = int(state.get("vision_issue_count", 0) or 0)
    storyboard_present = bool(state.get("storyboard_present"))

    if severity == "medium":
        if storyboard_present and retry_count >= 3 and issue_count <= 4:
            return _success_patch("Accepted medium-severity visual issue with storyboard guidance.")
        if retry_count >= 4 and issue_count <= 2:
            return _success_patch("Accepted medium-severity visual issue after max retries.")

    if state.get("vision_error"):
        return _failure_patch(
            verdict="content_failure", stage="vision", failure_type="content",
            reason=str(state.get("vision_error") or "Vision review requested revision."),
        )

    if state.get("render_error"):
        return _failure_patch(
            verdict="content_failure", stage="execution", failure_type="content",
            reason=str(state.get("render_error")),
        )

    if state.get("ast_error"):
        return _failure_patch(
            verdict="content_failure", stage="ast", failure_type="content",
            reason=str(state.get("ast_error")),
        )

    return _failure_patch(
        verdict="content_failure", stage="unknown", failure_type="unknown",
        reason="Workflow ended without a definitive result.",
    )


def build_plan_graph() -> object:
    """Build planner-only graph for Planning mode."""

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)
    return graph.compile()


def build_generate_graph() -> object:
    """Build generation graph for code/render/vision closed loop."""

    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("coder", coder_node)
    graph.add_node("ast_reviewer", ast_reviewer_node)
    graph.add_node("execution", execution_node)
    graph.add_node("vision_critic", vision_critic_node)
    graph.add_node("verdict", verdict_node)

    graph.set_entry_point("coder")
    graph.add_edge("coder", "ast_reviewer")

    graph.add_conditional_edges(
        "ast_reviewer",
        route_after_ast,
        {
            "to_coder": "coder",
            "to_execution": "execution",
            "finish": "verdict",
        },
    )

    graph.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "to_coder": "coder",
            "to_vision": "vision_critic",
            "finish": "verdict",
        },
    )

    graph.add_conditional_edges(
        "vision_critic",
        route_after_vision,
        {
            "to_coder": "coder",
            "finish": "verdict",
        },
    )

    graph.add_edge("verdict", END)

    return graph.compile()


plan_only_app = build_plan_graph()
generate_app = build_generate_graph()
