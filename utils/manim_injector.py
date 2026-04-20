from __future__ import annotations

import re

INJECTION_MARKER = "# __AUTO_ANCHOR_INJECTED__"


def _leading_ws(line: str) -> str:
    """Return leading whitespace from one line."""

    match = re.match(r"^\s*", line)
    return match.group(0) if match else ""


def _build_injection(indent: str) -> str:
    """Build non-recursive temporary anchor snippet."""

    return "\n".join(
        [
            f"{indent}{INJECTION_MARKER}",
            f"{indent}if self.mobjects:",
            (
                f"{indent}    _bbox_group = VGroup(*["
                "SurroundingRectangle(m, color=RED, buff=0.1) for m in self.mobjects"
                "])"
            ),
            f"{indent}    self.add(_bbox_group)",
            f"{indent}    self.wait(0.01)",
            f"{indent}    self.remove(_bbox_group)",
        ]
    )


def inject_bounding_boxes(code: str) -> str:
    """
    Inject temporary red boxes before each self.play() call.

    Pure string/regex implementation.
    """

    if not code.strip():
        return code
    if INJECTION_MARKER in code:
        return code

    lines = code.splitlines()
    play_pattern = re.compile(r"^\s*self\.play\s*\(")
    output: list[str] = []
    injected_any = False

    for line in lines:
        if play_pattern.search(line):
            indent = _leading_ws(line)
            output.append(_build_injection(indent))
            injected_any = True
        output.append(line)

    if not injected_any:
        output.append("")
        output.append(_build_injection("        "))

    merged = "\n".join(output)
    if code.endswith("\n"):
        return f"{merged}\n"
    return merged

