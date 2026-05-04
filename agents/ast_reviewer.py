from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from state import AgentState

BLOCKED_IMPORTS: set[str] = {
    "asyncio",
    "builtins",
    "ctypes",
    "ftplib",
    "glob",
    "httpx",
    "importlib",
    "marshal",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "ssl",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "urllib",
}

BLOCKED_CALLS: set[str] = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "rmtree",
    "os.system",
    "os.popen",
    "os.remove",
    "os.unlink",
    "setattr",
    "shutil.rmtree",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
}

BLOCKED_CALL_SUFFIXES: tuple[str, ...] = (
    ".read_text",
    ".write_text",
    ".read_bytes",
    ".write_bytes",
    ".open",
    ".unlink",
    ".rmdir",
    ".mkdir",
    ".rename",
    ".glob",
    ".rglob",
    ".iterdir",
    ".rmtree",
)

RUNTIME_ONLY_STRATEGY = "Runtime Only"


def _resolve_call_name(node: ast.AST) -> str:
    """Resolve dotted name from ast node, e.g., os.system."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _resolve_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


@dataclass
class _SafetyVisitor(ast.NodeVisitor):
    """AST visitor collecting contract and security signals."""

    has_generated_scene: bool = False
    has_construct: bool = False
    blocked_imports: list[str] = field(default_factory=list)
    blocked_calls: list[str] = field(default_factory=list)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BLOCKED_IMPORTS:
                self.blocked_imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".")[0]
        if module in BLOCKED_IMPORTS:
            self.blocked_imports.append(node.module or "")
        full_module = node.module or ""
        if full_module.startswith("manim.") and full_module != "manim":
            self.blocked_imports.append(full_module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == "GeneratedScene":
            self.has_generated_scene = True
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "construct":
                    self.has_construct = True
                    break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _resolve_call_name(node.func)
        if (
            call_name in BLOCKED_CALLS
            or call_name.startswith("subprocess.")
            or call_name.startswith("os.")
            or any(call_name.endswith(suffix) for suffix in BLOCKED_CALL_SUFFIXES)
        ):
            self.blocked_calls.append(call_name)
        self.generic_visit(node)


def _format_syntax_error(error: SyntaxError) -> str:
    """Build concise syntax error text."""

    line = error.lineno or 0
    col = error.offset or 0
    msg = error.msg or "Invalid Python syntax."
    return f"SyntaxError at line {line}, column {col}: {msg}"


def ast_reviewer_node(state: AgentState) -> dict[str, Any]:
    """Static review node for syntax, safety, and class contract."""

    if state.get("strategy") == RUNTIME_ONLY_STRATEGY:
        return {"ast_error": None, "ast_error_ratio": 0.0}

    try:
        code = str(state.get("code", ""))
    except Exception:
        code = ""

    if not code.strip():
        return {"ast_error": "Generated code is empty."}

    try:
        tree = ast.parse(code)
        # 计算总节点数
        total_nodes = len(list(ast.walk(tree)))
    except Exception as exc:
        return {
            "ast_error": f"AST parse failed: {exc}",
            "ast_error_ratio": 1.0 # 解析失败，畸变率最高
        }

    # 使用你现有的 visitor 检查错误
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    
    # 计算错误节点权重 (你可以根据需要调整权重逻辑)
    error_count = len(visitor.blocked_imports) + len(visitor.blocked_calls)
    if not visitor.has_generated_scene or not visitor.has_construct:
        error_count += 2
        
    # 计算最终比率
    ast_ratio = min(1.0, error_count / total_nodes) if total_nodes > 0 else 0.0

    # 获取现有的错误信息
    error_msg = None
    if not visitor.has_generated_scene:
        error_msg = "Missing required class: GeneratedScene."
    elif not visitor.has_construct:
        error_msg = "GeneratedScene must define construct(self)."
    # 👇 把之前被注释掉的安全拦截逻辑补回来
    elif visitor.blocked_imports:
        error_msg = f"Blocked imports detected: {', '.join(visitor.blocked_imports)}"
    elif visitor.blocked_calls:
        error_msg = f"Blocked function calls detected: {', '.join(visitor.blocked_calls)}"

    return {
        "ast_error": error_msg,
        "ast_error_ratio": ast_ratio # 将算出的比率写回状态
    }
