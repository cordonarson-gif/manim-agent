from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from state import create_initial_state

DATASET_PATH = "hf://datasets/SuienR/ManimBench-v1/manim_sft_dataset_test_v2.parquet"
OUTPUT_CSV = "experiment_results_full.csv"
STRATEGIES = ["Runtime Only", "Ours"]
RUNTIME_ONLY_STRATEGY = "Runtime Only"
OURS_STRATEGY = "Ours"
SELECTED_TASK_IDS: list[int] = []
RANDOM_SAMPLE_SIZE = 2
RANDOM_STATE = 42
MAX_API_RETRIES = 3
RETRY_SLEEP_SECONDS = 10


def _configure_cli_logging() -> None:
    """Enable node-level progress logs for CLI experiment runs."""

    level_name = os.getenv("MANIM_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _merge_state(runtime_state: dict[str, Any], patch: dict[str, Any] | None) -> dict[str, Any]:
    """Merge graph output back into the working state."""

    if isinstance(patch, dict):
        runtime_state.update(patch)
    return runtime_state


def _normalize_optional_text(value: Any) -> str | None:
    """Normalize optional error text for consistent success checks."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _enabled_strategies() -> list[str]:
    """Return experiment strategies in a stable comparison order."""

    return list(STRATEGIES)


def _select_test_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    """Pick a reproducible experiment slice."""

    if SELECTED_TASK_IDS:
        available_ids = [task_id for task_id in SELECTED_TASK_IDS if task_id in df.index]
        missing_ids = [task_id for task_id in SELECTED_TASK_IDS if task_id not in df.index]
        if missing_ids:
            print(f"⚠️ 数据集中缺少 task_id: {missing_ids}")
        if not available_ids:
            raise ValueError("Configured task ids are not present in the dataset.")
        return df.loc[available_ids]

    return df.sort_index()


def _has_valid_storyboard(value: Any) -> bool:
    """Return whether storyboard text is present and non-empty."""

    return isinstance(value, str) and bool(value.strip())


def _run_strategy(task: str, strategy: str) -> dict[str, Any]:
    """Run one strategy end-to-end and return the final merged state."""

    from workflow import generate_app, plan_only_app

    runtime_state = dict(create_initial_state(task=task, strategy=strategy))

    if strategy == OURS_STRATEGY:
        planning_result = plan_only_app.invoke(runtime_state)
        _merge_state(runtime_state, planning_result)
        runtime_state["planner_used"] = True
        runtime_state["storyboard_present"] = _has_valid_storyboard(runtime_state.get("storyboard"))
        if not runtime_state["storyboard_present"]:
            runtime_state.update(
                {
                    "is_success": 0,
                    "final_verdict": "content_failure",
                    "failure_stage": "planner",
                    "failure_type": "content",
                    "failure_reason": "Ours strategy requires a valid storyboard before generation.",
                    "success_reason": None,
                }
            )
            return runtime_state
    else:
        runtime_state["planner_used"] = False
        runtime_state["storyboard"] = None
        runtime_state["storyboard_present"] = False

    generation_result = generate_app.invoke(runtime_state)
    _merge_state(runtime_state, generation_result)
    runtime_state["storyboard_present"] = _has_valid_storyboard(runtime_state.get("storyboard"))
    return runtime_state


def main() -> int:
    """Run the configured ManimBench experiment slice."""

    import pandas as pd

    _configure_cli_logging()
    print("📦 正在读取 ManimBench 测试集...")
    df = pd.read_parquet(DATASET_PATH)
    df_test = _select_test_rows(df)
    strategies = _enabled_strategies()

    experiment_results: list[dict[str, Any]] = []
    print(f"🚀 开始执行自动化评测，共 {len(df_test)} 个测试用例...")
    print(f"🔒 启用策略: {', '.join(strategies)}")

    for task_id, row in df_test.iterrows():
        instruction = str(row["Reviewed Description"])

        for strategy in strategies:
            print("\n========================================")
            print(f"▶️ 正在处理 Task {task_id} | 策略: {strategy}")
            print("========================================")
            start_time = time.time()

            api_retry_count = 0
            final_state: dict[str, Any] | None = None

            while api_retry_count < MAX_API_RETRIES:
                try:
                    final_state = _run_strategy(instruction, strategy)
                    break
                except Exception as exc:
                    api_retry_count += 1
                    print(f"⚠️ 遇到报错: {exc}")
                    if api_retry_count < MAX_API_RETRIES:
                        print(
                            f"🔁 正在尝试重新连接 "
                            f"({api_retry_count}/{MAX_API_RETRIES})，休息 {RETRY_SLEEP_SECONDS} 秒..."
                        )
                        time.sleep(RETRY_SLEEP_SECONDS)
                    else:
                        print(f"❌ 任务 {task_id} 在策略 {strategy} 下连续失败，已跳过。")

            if final_state:
                ast_error = _normalize_optional_text(final_state.get("ast_error"))
                render_error = _normalize_optional_text(final_state.get("render_error"))
                vision_error = _normalize_optional_text(final_state.get("vision_error"))
                iteration_count = int(final_state.get("retry_count", 0) or 0)
                ast_error_ratio = float(final_state.get("ast_error_ratio", 0.0) or 0.0)
                vlm_iou_score = float(final_state.get("vlm_iou_score", 0.0) or 0.0)
                render_image_path = final_state.get("render_image_path")
                render_video_path = final_state.get("render_video_path")
                final_verdict = str(final_state.get("final_verdict", "unknown") or "unknown")
                failure_stage = _normalize_optional_text(final_state.get("failure_stage"))
                failure_type = _normalize_optional_text(final_state.get("failure_type"))
                failure_reason = _normalize_optional_text(final_state.get("failure_reason"))
                success_reason = _normalize_optional_text(final_state.get("success_reason"))
                execution_environment = _normalize_optional_text(final_state.get("execution_environment"))
                docker_context = _normalize_optional_text(final_state.get("docker_context"))
                vision_skipped = int(bool(final_state.get("vision_skipped", False)))
                vision_verdict = _normalize_optional_text(final_state.get("vision_verdict"))
                vision_severity = _normalize_optional_text(final_state.get("vision_severity"))
                vision_issue_count = int(final_state.get("vision_issue_count", 0) or 0)
                planner_used = int(bool(final_state.get("planner_used", False)))
                storyboard_present = int(bool(final_state.get("storyboard_present", False)))
                coder_storyboard_used = int(bool(final_state.get("coder_storyboard_used", False)))
                coder_input_mode = _normalize_optional_text(final_state.get("coder_input_mode"))
                retry_count_reason = _normalize_optional_text(final_state.get("retry_count_reason"))
                is_success = 1 if final_verdict == "success" else 0
            else:
                ast_error = "Workflow invocation failed after API retries."
                render_error = None
                vision_error = None
                iteration_count = 0
                ast_error_ratio = 1.0
                vlm_iou_score = 1.0
                render_image_path = None
                render_video_path = None
                final_verdict = "infra_failure"
                failure_stage = "workflow"
                failure_type = "infra"
                failure_reason = ast_error
                success_reason = None
                execution_environment = None
                docker_context = None
                vision_skipped = int(strategy == RUNTIME_ONLY_STRATEGY)
                vision_verdict = None
                vision_severity = None
                vision_issue_count = 0
                planner_used = int(strategy == OURS_STRATEGY)
                storyboard_present = 0
                coder_storyboard_used = 0
                coder_input_mode = None
                retry_count_reason = "coder attempt count"
                is_success = 0

            time_cost = time.time() - start_time

            experiment_results.append(
                {
                    "task_id": task_id,
                    "strategy": strategy,
                    "planner_used": planner_used,
                    "storyboard_present": storyboard_present,
                    "coder_storyboard_used": coder_storyboard_used,
                    "coder_input_mode": coder_input_mode,
                    "retry_count_reason": retry_count_reason,
                    "iteration_count": iteration_count,
                    "is_success": is_success,
                    "final_verdict": final_verdict,
                    "failure_stage": failure_stage,
                    "failure_type": failure_type,
                    "failure_reason": failure_reason,
                    "success_reason": success_reason,
                    "execution_environment": execution_environment,
                    "docker_context": docker_context,
                    "vision_skipped": vision_skipped,
                    "vision_verdict": vision_verdict,
                    "vision_severity": vision_severity,
                    "vision_issue_count": vision_issue_count,
                    "ast_error_ratio": ast_error_ratio,
                    "vlm_iou_score": vlm_iou_score,
                    "ast_error": ast_error,
                    "render_error": render_error,
                    "vision_error": vision_error,
                    "render_image_path": render_image_path,
                    "render_video_path": render_video_path,
                    "time_cost": time_cost,
                }
            )

            print(
                f"✅ 完成！耗时 {time_cost:.1f}s | 成功: {is_success} | "
                f"结论: {final_verdict} | 迭代: {iteration_count}"
            )

    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 小批量跑批完成！已生成 {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
