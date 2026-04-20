# manim-agent

An EI-grade dual-feedback pipeline for automated Manim teaching content generation.

`manim-agent` is a multi-agent workflow built on LangGraph for generating, reviewing, rendering, and revising Manim teaching animations from natural-language prompts.

## Overview

This project splits Manim content generation into several specialized agents and lets them collaborate in a closed loop:

- `planner`: turns a teaching request into storyboard JSON
- `coder`: generates or repairs Manim code
- `ast_reviewer`: checks syntax, safety, and scene contract
- `execution`: renders image/video artifacts in a sandboxed run directory
- `vision_critic`: reviews the rendered frame and returns visual feedback

The system supports both planning-only and end-to-end generation workflows, with a hard retry cap of `MAX_RETRIES = 5`.

## Architecture

```text
User Task
   |
   v
planner (optional, planning mode)
   |
   v
coder
   |
   v
ast_reviewer
   | pass
   v
execution
   | pass
   v
vision_critic
   | pass
   v
finish

If AST / render / vision review fails:
coder <- feedback <- ast_reviewer / execution / vision_critic
```

## Features

- Dual-feedback repair loop across code-level review and visual review
- Planning mode for storyboard generation only
- Generation mode with iterative repair and render validation
- CLI entrypoint and Chainlit chat interface
- Per-run render artifacts under `media/runs`
- Lightweight experiment logging under `logs/runs`

## Project Structure

```text
.
|-- app.py
|-- main.py
|-- workflow.py
|-- state.py
|-- requirements.txt
|-- agents/
|   |-- planner.py
|   |-- coder.py
|   |-- ast_reviewer.py
|   |-- execution.py
|   `-- vision_critic.py
`-- utils/
    |-- experiment_logger.py
    |-- manim_injector.py
    `-- model_provider.py
```

## Download

After the repository is published to GitHub, clone it with:

```bash
git clone https://github.com/<your-github-username>/manim-agent.git
cd manim-agent
```

You can also download a ZIP package from the GitHub repository page:

`Code -> Download ZIP`

## Requirements

- Python 3.10+
- Manim Community Edition 0.19+
- FFmpeg available in `PATH`
- API access for:
  - DeepSeek chat model
  - Qwen vision model

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set the required environment variables before running the project.

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
$env:QWEN_API_KEY="your_qwen_api_key"
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

macOS / Linux:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export QWEN_API_KEY="your_qwen_api_key"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

Optional model overrides:

```bash
export MANIM_PLANNER_MODEL=deepseek-chat
export MANIM_CODER_MODEL=deepseek-chat
export MANIM_VISION_MODEL=qwen-vl-max
```

Optional execution tuning:

```bash
export MANIM_EXEC_TIMEOUT=120
export MANIM_ERROR_LIMIT=1800
```

## Usage

### 1. Planning mode

Use the `planner` agent only and output storyboard JSON.

### 2. Generation mode

If a storyboard already exists:

```text
coder -> ast_reviewer -> execution -> vision_critic
```

If no storyboard exists, the planner runs first and then enters the same generation loop.

### CLI

```bash
python main.py --task "Explain how a triangle transforms into a square"
```

Quiet mode:

```bash
python main.py --task "Explain Pythagorean theorem visually" --quiet
```

### Chainlit UI

```bash
chainlit run app.py
```

Note: for long rendering sessions, avoid watch mode because temporary file changes may trigger reloads unexpectedly.

## Outputs

- Render artifacts: `media/runs`
- Latest Manim outputs may also appear under `media/images` and `media/videos`
- Experiment logs: `logs/runs/*.jsonl`

These folders are runtime artifacts and are normally excluded from Git version control.

## Tech Stack

- LangGraph
- LangChain
- ChatOpenAI-compatible model clients
- Manim Community Edition
- Chainlit

## Notes

- This repository has been prepared for GitHub publishing by removing hardcoded API secrets.
- Please keep all model credentials in environment variables and do not commit them into source files.
- Some files in the current workspace are local runtime outputs and are intentionally ignored when publishing.

## License

No license file is included yet. Add one before wider open-source distribution if needed.
