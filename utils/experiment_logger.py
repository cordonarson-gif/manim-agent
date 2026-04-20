from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Lightweight JSONL logger for reproducible workflow experiments."""

    def __init__(self, root_dir: Path | None = None, run_id: str | None = None) -> None:
        self.run_id: str = run_id or f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.root_dir: Path = root_dir or Path("logs/runs")
        self.file_path: Path = self.root_dir / f"{self.run_id}.jsonl"

        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Logging must never break the main workflow.
            pass

    def log(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append one structured event line."""

        event = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event_type": event_type,
            "payload": payload or {},
        }
        try:
            line = json.dumps(event, ensure_ascii=False)
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(f"{line}\n")
        except Exception:
            # Best-effort only.
            return

