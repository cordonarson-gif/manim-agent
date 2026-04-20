from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from agents.ast_reviewer import ast_reviewer_node
from agents.coder import coder_node
from agents.execution import execution_node
from agents.planner import planner_node
from agents.vision_critic import vision_critic_node
from state import AgentState, MAX_RETRIES

GenerateRoute = Literal["to_coder", "to_execution", "to_vision", "finish"]


def _should_force_end(state: AgentState) -> bool:
    """Hard-stop guard to prevent infinite retry loops."""

    try:
        return int(state.get("retry_count", 0)) >= MAX_RETRIES
    except Exception:
        return True


def _is_vision_passed(vision_error: str | None) -> bool:
    """Vision stage passes when feedback is None or explicit OK."""

    if vision_error is None:
        return True
    return vision_error.strip().upper() == "OK"


def _is_minor_vision_issue(vision_error: str | None) -> bool:
    """Best-effort detector for low-severity vision feedback."""

    if vision_error is None:
        return False
    text = vision_error.strip().lower()
    if not text:
        return False
    minor_tokens = ["severity\": \"low", "severity: low", "minor", "轻微", "low"]
    return any(token in text for token in minor_tokens)


def route_after_ast(state: AgentState) -> GenerateRoute:
    """Route after AST review."""

    if state.get("ast_error"):
        if _should_force_end(state):
            return "finish"
        return "to_coder"
    return "to_execution"


def route_after_execution(state: AgentState) -> GenerateRoute:
    """Route after execution sandbox."""

    if state.get("render_error"):
        if _should_force_end(state):
            return "finish"
        return "to_coder"
    return "to_vision"


def route_after_vision(state: AgentState) -> GenerateRoute:
    """Route after vision critic."""

    vision_error = state.get("vision_error")
    if _is_vision_passed(vision_error):
        return "finish"
    if _should_force_end(state) and _is_minor_vision_issue(vision_error):
        return "finish"
    if _should_force_end(state):
        return "finish"
    return "to_coder"


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

    graph.set_entry_point("coder")
    graph.add_edge("coder", "ast_reviewer")

    graph.add_conditional_edges(
        "ast_reviewer",
        route_after_ast,
        {
            "to_coder": "coder",
            "to_execution": "execution",
            "finish": END,
        },
    )

    graph.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "to_coder": "coder",
            "to_vision": "vision_critic",
            "finish": END,
        },
    )

    graph.add_conditional_edges(
        "vision_critic",
        route_after_vision,
        {
            "to_coder": "coder",
            "finish": END,
        },
    )

    return graph.compile()


plan_only_app = build_plan_graph()
generate_app = build_generate_graph()
