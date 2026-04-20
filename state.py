from __future__ import annotations

from typing import TypedDict

MAX_RETRIES: int = 5

class AgentState(TypedDict):
    """Global workflow state shared by all LangGraph nodes."""

    task: str
    storyboard: str | None
    code: str
    ast_error: str | None
    render_error: str | None
    vision_error: str | None
    retry_count: int
    render_media_dir: str | None
    render_image_path: str | None
    render_video_path: str | None
    # === 新增：为论文数据分析准备的“出水管” ===
    ast_error_ratio: float  # 记录 AST 畸变率
    vlm_iou_score: float    # 记录 VLM 视觉重叠度
    is_success: int         # 记录最终是否修复成功 (1或0)

def create_initial_state(task: str = "") -> AgentState:
    """Create a fully initialized state object."""

    return AgentState(
        task=task,
        storyboard=None,
        code="",
        ast_error=None,
        render_error=None,
        vision_error=None,
        retry_count=0,
        render_media_dir=None,
        render_image_path=None,
        render_video_path=None,
        # 👇 别忘了在这里给它们赋上初始值
        ast_error_ratio=0.0,
        vlm_iou_score=0.0,
        is_success=0,
    )