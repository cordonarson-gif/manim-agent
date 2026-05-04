from __future__ import annotations

from typing import Literal, TypedDict

MAX_RETRIES: int = 5
FinalVerdict = Literal["success", "content_failure", "infra_failure", "unknown"]
FailureStage = Literal["ast", "execution", "vision", "planner", "workflow"]
FailureType = Literal["content", "infra", "unknown"]
ExecutionEnvironment = Literal["docker", "local", "unknown"]
CoderInputMode = Literal["storyboard_json", "natural_language", "unknown"]


class AgentState(TypedDict):
    """Global workflow state shared by all LangGraph nodes."""

    task: str
    strategy: str
    storyboard: str | None
    code: str
    ast_error: str | None
    render_error: str | None
    vision_error: str | None
    retry_count: int
    render_media_dir: str | None
    render_image_path: str | None
    render_video_path: str | None
    ast_error_ratio: float
    vlm_iou_score: float
    is_success: int
    execution_environment: ExecutionEnvironment
    docker_context: str | None
    failure_stage: FailureStage | None
    failure_type: FailureType | None
    final_verdict: FinalVerdict
    success_reason: str | None
    failure_reason: str | None
    vision_skipped: bool
    vision_verdict: str | None
    vision_severity: str | None
    vision_issue_count: int
    planner_used: bool
    storyboard_present: bool
    coder_input_mode: CoderInputMode
    coder_storyboard_used: bool
    retry_count_reason: str


def create_initial_state(task: str = "", strategy: str = "Ours") -> AgentState:
    """Create a fully initialized state object."""

    return AgentState(
        task=task,
        strategy=strategy,
        storyboard=None,
        code="",
        ast_error=None,
        render_error=None,
        vision_error=None,
        retry_count=0,
        render_media_dir=None,
        render_image_path=None,
        render_video_path=None,
        ast_error_ratio=0.0,
        vlm_iou_score=0.0,
        is_success=0,
        execution_environment="unknown",
        docker_context=None,
        failure_stage=None,
        failure_type=None,
        final_verdict="unknown",
        success_reason=None,
        failure_reason=None,
        vision_skipped=False,
        vision_verdict=None,
        vision_severity=None,
        vision_issue_count=0,
        planner_used=False,
        storyboard_present=False,
        coder_input_mode="unknown",
        coder_storyboard_used=False,
        retry_count_reason="coder attempt count",
    )
