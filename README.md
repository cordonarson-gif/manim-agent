# manim-agent

> Multi-agent dual-feedback pipeline for automated Manim teaching content generation.  
> 面向 Manim 教学动画自动生成的多智能体双反馈系统。

`manim-agent` is a LangGraph-based workflow that turns natural-language teaching requests into storyboard plans, Manim code, rendered media, and visual feedback-driven revisions. It is designed for educational animation generation with a closed repair loop across code review and vision review.

`manim-agent` 是一个基于 LangGraph 的多智能体工作流，能够将自然语言教学需求转换为分镜规划、Manim 代码、渲染结果，并通过代码审查与视觉审查闭环迭代修复，更适合教学动画生成场景。

## Highlights | 项目亮点

- `Dual feedback loop`: combines AST/static review and visual critique to improve generation reliability.
- `双反馈闭环`：同时利用 AST 静态审查和视觉审查，提高动画生成的可执行性与可用性。

- `Planner-first workflow`: supports planning-only mode and end-to-end generation mode.
- `规划优先工作流`：同时支持仅输出分镜的 Planning 模式与完整生成的 Generation 模式。

- `Agent specialization`: planner, coder, AST reviewer, execution sandbox, and vision critic each focus on one stage.
- `智能体分工明确`：规划、代码生成、静态检查、执行渲染、视觉评估各司其职。

- `Retry-controlled pipeline`: hard stop at `MAX_RETRIES = 5` to avoid endless loops.
- `受控重试机制`：通过 `MAX_RETRIES = 5` 限制迭代次数，避免无限循环。

- `Usable interfaces`: supports both CLI usage and Chainlit web interaction.
- `可直接使用`：同时提供命令行入口和 Chainlit 交互界面。

## Workflow | 工作流

### Agent roles | 智能体角色

- `planner`: converts a teaching request into storyboard JSON.
- `planner`：将教学需求拆解为分镜 JSON。

- `coder`: generates or repairs Manim code from the task, storyboard, and feedback.
- `coder`：根据任务、分镜和反馈生成或修复 Manim 代码。

- `ast_reviewer`: checks syntax, safety rules, and `GeneratedScene` contract.
- `ast_reviewer`：检查语法、安全限制以及 `GeneratedScene` 约束。

- `execution`: renders snapshots and videos in an isolated run directory.
- `execution`：在独立运行目录中完成截图和视频渲染。

- `vision_critic`: reviews the rendered frame and returns visual/layout feedback.
- `vision_critic`：分析渲染图像并返回视觉与布局反馈。

### Architecture

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

### Core modes | 核心模式

- `Planning mode`: runs `planner` only and outputs storyboard JSON.
- `Planning 模式`：仅运行 `planner`，输出分镜 JSON。

- `Generation mode`: runs `coder -> ast_reviewer -> execution -> vision_critic`.
- `Generation 模式`：运行 `coder -> ast_reviewer -> execution -> vision_critic` 闭环。

- If no storyboard exists, the planner runs first and then enters the generation loop.
- 如果没有现成分镜，系统会先规划，再进入生成与修复循环。

## Quick Start | 快速开始

### Clone | 下载

```bash
git clone https://github.com/cordonarson-gif/manim-agent.git
cd manim-agent
```

You can also use `Code -> Download ZIP` on the GitHub repository page.  
也可以在 GitHub 仓库页面使用 `Code -> Download ZIP` 直接下载。

### Requirements | 环境要求

- Python 3.10+
- Manim Community Edition 0.19+
- FFmpeg available in `PATH`
- API access for DeepSeek chat and Qwen vision models

### Install | 安装

```bash
pip install -r requirements.txt
```

### Configuration | 环境变量配置

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

## Usage | 使用方式

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

For long rendering sessions, avoid watch mode because temporary file changes may trigger reloads unexpectedly.  
对于长时间渲染任务，建议避免 watch 模式，因为临时文件变化可能触发热重载。

## Project Structure | 项目结构

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

## Outputs | 输出结果

- Render artifacts: `media/runs`
- Snapshot or video outputs may also appear in `media/images` and `media/videos`
- Experiment logs: `logs/runs/*.jsonl`

These folders are runtime artifacts and are normally excluded from version control.  
这些目录属于运行产物，通常不会纳入 Git 版本管理。

## Tech Stack

- LangGraph
- LangChain
- ChatOpenAI-compatible model clients
- Manim Community Edition
- Chainlit

## Notes | 说明

- Hardcoded API keys have been removed before publishing.
- 仓库公开前已经移除了硬编码 API Key。

- Keep credentials in environment variables instead of source files.
- 建议将模型密钥保存在环境变量中，而不是源码中。

- Some local runtime outputs are intentionally ignored by `.gitignore`.
- 一些本地产生的运行结果已经通过 `.gitignore` 排除。

## License

No license file is included yet. Add one before wider open-source distribution if needed.  
当前仓库暂未附带许可证文件，如计划公开分发，建议补充 `LICENSE`。
