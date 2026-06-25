#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harness Kernel - Python Micro-code Representation

Changes from v1:
  - All config loaded from YAML (Pydantic models)
  - Skills are configurable via YAML, not hardcoded
  - Significant code simplification
  - PermissionManager integrated directly

Usage Example:
  # Simple query
  python3 run.py "Hello, who are you?"

  # Enable plan mode and ask
  python3 run.py --root .
  > /plan
  > Create a new file named test.txt with content 'hello'
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import os

os.environ["LITELLM_SKIP_HTTP_REQUESTS"] = "True"
import sys
import json
import re
import glob
import subprocess

if sys.platform != "win32":
    import readline
import urllib.request
import logging
import yaml
from loguru import logger
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from typing import List as TypingList
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.markdown import Markdown

# Initialize console
console = Console()


def rich_print(text: str):
    console.print(Markdown(text))


def _load_dotenv(env_file: str = ".env") -> None:
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:
        raise ImportError("Required dependency 'python-dotenv' is missing. Please install it with: pip install python-dotenv")


def _stdout(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


class AgentConfig(BaseModel):
    system_prompt: str = ""
    language_policy: str = ""
    max_steps: int = 0
    planner_max_steps: int = 12
    temperature: float = 0.0
    auto_plan: bool = False
    reasoning_language: str = "auto"
    planner_model: str = ""
    subagent_model: str = ""
    subagent_models: Dict[str, str] = Field(default_factory=dict)
    output_style: str = ""
    compact_ratio: float = 0.8
    compact_force_ratio: float = 0.9


class ShellConfig(BaseModel):
    prefer: str = "auto"
    path: str = ""


class ToolsConfig(BaseModel):
    enabled: List[str] = Field(default_factory=list)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    bash_timeout_seconds: int = 120


class SkillsConfig(BaseModel):
    enabled: List[str] = Field(default_factory=list)


class PermissionsConfig(BaseModel):
    allow_write: bool = True
    allow_bash: bool = True
    allow_web_fetch: bool = True
    auto_approve: bool = False


class SandboxConfig(BaseModel):
    enabled: bool = False
    allowed_paths: List[str] = Field(default_factory=list)
    blocked_paths: List[str] = Field(default_factory=list)


class ProviderEntry(BaseModel):
    name: str = ""
    kind: str = "openai"
    base_url: str = ""
    model: str = ""
    models: List[str] = Field(default_factory=list)
    default: bool = False
    api_key_env: str = ""
    context_window: int = 0
    request_timeout: int = 120
    delay_seconds: float = 0.0
    retry_times: int = 3
    price: Dict[str, float] = Field(default_factory=dict)
    effort: str = ""
    thinking: str = ""


class SkillEntry(BaseModel):
    """Skill definition from YAML — replaces hardcoded BUILTIN_SKILLS."""

    name: str
    description: str = ""
    body: str = ""
    path: str = ""
    allowed_tools: List[str] = Field(default_factory=list)
    run_as: str = "subagent"


class Config(BaseModel):
    """Root configuration model — loaded from YAML."""

    default_model: str = ""
    providers: List[ProviderEntry] = Field(default_factory=list)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    plan_mode_marker: str = ""
    plan_approved_message: str = ""
    # skills_data is loaded separately from skills.yaml, not from main config

    @classmethod
    def load_for_root(cls, workspace_root: str) -> Config:
        """Load config from YAML with resolution: project > user > defaults."""
        cfg = Config()

        user_config = Path.home() / ".reasonix" / "config.yaml"
        if user_config.exists():
            cfg = cfg._merge_yaml(user_config)

        project_config = Path(workspace_root) / "config.yaml"
        if project_config.exists():
            cfg = cfg._merge_yaml(project_config)

        skills_yaml = Path(workspace_root) / "skills.yaml"
        all_skills = []
        if skills_yaml.exists():
            with open(skills_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                all_skills.extend([SkillEntry.model_validate(s) for s in data.get("skills", [])])

        skills_dir = Path(workspace_root) / "skills"
        if skills_dir.exists():
            for s_dir in skills_dir.iterdir():
                if not s_dir.is_dir():
                    continue
                md_path = s_dir / "SKILL.md"
                if not md_path.exists():
                    continue
                content = md_path.read_text(encoding="utf-8")
                data = {"name": s_dir.name, "body": content, "path": str(s_dir.resolve())}
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}
                        if isinstance(fm, dict):
                            data["name"] = fm.get("name", s_dir.name)
                            data["description"] = fm.get("description", "")
                all_skills.append(SkillEntry.model_validate(data))

        cfg._skills_data = all_skills
        return cfg

    def _merge_yaml(self, path: Path) -> Config:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        current = self.model_dump()
        merged = _deep_merge(current, data)
        return Config.model_validate(merged)

    @staticmethod
    def _resolve_text_or_file(value: str, root: str) -> str:
        """If value starts with 'file:', read the referenced file (relative to root).

        Supported formats:
          system_prompt: "file:prompt.md"          # relative to workspace root
          system_prompt: "file:./prompts/sys.md"   # same, explicit ./
          system_prompt: "file:/abs/path/sys.md"   # absolute path
        If the prefix is absent the value is returned as-is.
        """
        stripped = value.strip()
        if not stripped.lower().startswith("file:"):
            return value
        file_path_str = stripped[5:].strip()
        file_path = Path(file_path_str)
        if not file_path.is_absolute():
            file_path = Path(root) / file_path
        try:
            return file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"system_prompt file not found: {file_path}")

    def resolve_system_prompt(self, root: str) -> str:
        base = self._resolve_text_or_file(self.agent.system_prompt, root)
        if self.agent.language_policy:
            base += "\n\n" + self._resolve_text_or_file(self.agent.language_policy, root)
        if self.agent.output_style:
            base += f"\n\nOutput style: {self.agent.output_style}"
        memory_path = Path(root) / "REASONIX.md"
        if memory_path.exists():
            base += f"\n\n# Project Memory\n{memory_path.read_text(encoding='utf-8')}"
        return base

    @property
    def skills_data(self) -> List[SkillEntry]:
        return getattr(self, "_skills_data", [])

    def get_skill(self, name: str) -> Optional[SkillEntry]:
        for s in self.skills_data:
            if s.name == name:
                return s
        return None

    def enabled_skills(self) -> List[SkillEntry]:
        if not self.skills.enabled:
            return self.skills_data
        return [s for s in self.skills_data if s.name in self.skills.enabled]


# =============================================================================
# Logging helpers
# =============================================================================

_LOG_STYLES = {
    "user": ("📝 USER", "│"),
    "model": ("🤖 MODEL", "│"),
    "tool": ("🔧 TOOL", "│"),
    "mcp": ("🌐 MCP", "│"),
    "boot": ("⚙️  BOOT", "│"),
}


def log_box(category: str, text: str, max_width: int = 0) -> None:
    """Print text in a left-bordered box with category label."""
    style = _LOG_STYLES.get(category, (category.upper(), "│"))
    label, bar = style
    lines = text.splitlines() or [""]
    if max_width > 0:
        lines = [l[:max_width] for l in lines]
    width = max(len(label) + 2, max(len(l) for l in lines) + 2, 40)
    _stdout(f"┌─ {label} {'─' * (width - len(label) - 3)}")
    for l in lines:
        _stdout(f"{bar} {l}")
    _stdout(f"└{'─' * (width)}")


# =============================================================================
# 2. Built-in Tools (simplified, self-registering)
# =============================================================================


def should_enable(tool_name: str, enabled_list: List[str]) -> bool:
    return not enabled_list or tool_name in enabled_list


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = value
        else:
            result[key] = value
    return result


class Tool(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def description(self) -> str: ...
    @abstractmethod
    def schema(self) -> dict: ...
    @abstractmethod
    def execute(self, ctx: Any, args: dict) -> str: ...
    @abstractmethod
    def read_only(self) -> bool: ...

    def to_dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.schema(),
            },
        }


class ReadFileTool(Tool):
    def name(self):
        return "read_file"

    def description(self):
        return "Read a file's contents."

    def schema(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def read_only(self):
        return True

    def execute(self, ctx, args):
        try:
            return Path(args["path"]).read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return f"Error: {e}"


class WriteFileTool(Tool):
    def name(self):
        return "write_file"

    def description(self):
        return "Write content to a file. Overwrites if exists."

    def schema(self):
        return {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}

    def read_only(self):
        return False

    def execute(self, ctx, args):
        if hasattr(ctx, "perm_manager") and not ctx.perm_manager.check_and_request_permission(ctx, args["path"]):
            return "Error: Permission denied by user."
        path = Path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Wrote {path} ({len(args['content'])} chars)"


class EditFileTool(Tool):
    def name(self):
        return "edit_file"

    def description(self):
        return "Edit a file with search/replace."

    def schema(self):
        return {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}

    def read_only(self):
        return False

    def execute(self, ctx, args):
        if hasattr(ctx, "perm_manager") and not ctx.perm_manager.check_and_request_permission(ctx, args["path"]):
            return "Error: Permission denied by user."
        path = Path(args["path"])
        content = path.read_text(encoding="utf-8")
        old = args["old_text"]
        if old not in content:
            return f"Error: old_text not found in {path}"
        path.write_text(content.replace(old, args["new_text"], 1), encoding="utf-8")
        return f"Edited {path}"


class MultiEditTool(Tool):
    def name(self):
        return "multi_edit"

    def description(self):
        return "Apply multiple edits to a file atomically."

    def schema(self):
        return {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {"old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["old_text", "new_text"]}}}, "required": ["path", "edits"]}

    def read_only(self):
        return False

    def execute(self, ctx, args):
        if hasattr(ctx, "perm_manager") and not ctx.perm_manager.check_and_request_permission(ctx, args["path"]):
            return "Error: Permission denied by user."
        path = Path(args["path"])
        content = path.read_text(encoding="utf-8")
        for edit in args["edits"]:
            old = edit["old_text"]
            if old not in content:
                return f"Error: old_text not found: {old[:50]}..."
            content = content.replace(old, edit["new_text"], 1)
        path.write_text(content, encoding="utf-8")
        return f"Applied {len(args['edits'])} edits to {path}"


class BashTool(Tool):
    def __init__(self, prefer="auto", path="", timeout=120):
        self.prefer, self.path, self.timeout = prefer, path, timeout

    def name(self):
        return "bash"

    def description(self):
        return "Execute a shell command. Use for builds, tests, git, package managers."

    def schema(self):
        return {"type": "object", "properties": {"command": {"type": "string"}, "run_in_background": {"type": "boolean"}}, "required": ["command"]}

    def read_only(self):
        return False

    def execute(self, ctx, args):
        try:
            result = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=self.timeout, executable=self.path or None)
            out = result.stdout
            if result.returncode != 0:
                logger.error(f"Bash command failed: {args['command']}\n{result.stderr}")
                out += f"\n[exit {result.returncode}]\n{result.stderr}"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            logger.error(f"Bash command timed out: {args['command']}")
            return f"Error: timed out after {self.timeout}s"
        except Exception as e:
            logger.error(f"Bash command error: {args['command']}: {e}")
            return f"Error: {e}"


class GrepTool(Tool):
    def __init__(self, rg_path=""):
        self.rg_path = rg_path

    def name(self):
        return "grep"

    def description(self):
        return "Search for a regex pattern in files."

    def schema(self):
        return {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]}

    def read_only(self):
        return True

    def execute(self, ctx, args):
        pattern, path = args["pattern"], Path(args["path"])
        try:
            if self.rg_path and Path(self.rg_path).exists():
                r = subprocess.run([self.rg_path, "--no-heading", "-n", "--with-filename", pattern, str(path)], capture_output=True, text=True)
                return r.stdout or "(no matches)"
            rx = re.compile(pattern)
            files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
            matches = []
            for fp in files:
                try:
                    for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if rx.search(line):
                            matches.append(f"{fp}:{i}:{line}")
                except Exception:
                    continue
            return "\n".join(matches) if matches else "(no matches)"
        except Exception as e:
            return f"Error: {e}"


class GlobTool(Tool):
    def name(self):
        return "glob"

    def description(self):
        return "Find files matching a glob pattern."

    def schema(self):
        return {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}

    def read_only(self):
        return True

    def execute(self, ctx, args):
        return "\n".join(glob.glob(args["pattern"], recursive=True)) or "(no matches)"


class LsTool(Tool):
    def name(self):
        return "ls"

    def description(self):
        return "List directory contents."

    def schema(self):
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def read_only(self):
        return True

    def execute(self, ctx, args):
        p = Path(args["path"])
        return "\n".join(f"{'d' if e.is_dir() else 'f'} {e.name}" for e in sorted(p.iterdir())) if p.exists() else f"Error: not found {p}"


class WebFetchTool(Tool):
    def __init__(self, proxy=None):
        self.proxy = proxy

    def name(self):
        return "web_fetch"

    def description(self):
        return "Fetch content from a URL."

    def schema(self):
        return {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

    def read_only(self):
        return True

    def execute(self, ctx, args):
        try:
            req = urllib.request.Request(args["url"], headers={"User-Agent": "Harness/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error: {e}"


class AskTool(Tool):
    def name(self):
        return "ask"

    def description(self):
        return "Ask the user for clarification when a consequential choice is required."

    def schema(self):
        return {"type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}, "required": ["question"]}

    def read_only(self):
        return True

    def execute(self, ctx, args):
        _stdout(f"\n[ASK] {args['question']}")
        for i, opt in enumerate(args.get("options", []), 1):
            _stdout(f"  {i}. {opt}")
        if not sys.stdin.isatty():
            return "<model-assumption> Proceeding with default."
        try:
            choice = input("Your choice: ")
            logger.info(f"User chose: {choice}")
            return choice
        except EOFError:
            return "<model-assumption> Proceeding with default."


class TodoWriteTool(Tool):
    def name(self):
        return "todo_write"

    def description(self):
        return "Track multi-step task progress."

    def schema(self):
        return {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "content", "status"]}}}, "required": ["todos"]}

    def read_only(self):
        return False

    def execute(self, ctx, args):
        target = ctx.context if hasattr(ctx, "context") else ctx
        if hasattr(target, "todos"):
            target.todos = args["todos"]
            # Visual feedback for task progress in terminal
            todo_display = "\n".join([f"  [{t['status']:<11}] {t['content']}" for t in args["todos"]])
            _stdout(f"\n\U0001f4dd TASK PROGRESS:\n{todo_display}\n")
        return f"Updated {len(args['todos'])} todos"


# =============================================================================
# Plan Mode support
# =============================================================================

# (Moved to config.yaml)


def parse_plan_todos(plan: str) -> List[dict]:
    """Extract a starter task list from an approved plan's markdown list items.

    Mirrors DeepSeek-Harness parsePlanTodos (internal/control/controller.go).
    First item gets status='in_progress'; the rest 'pending'. Capped at 20.
    """
    import re

    todos: List[dict] = []
    for raw in plan.splitlines():
        stripped = raw.lstrip(" \t")
        if not stripped:
            continue
        content: Optional[str] = None
        level = 0
        indent = len(raw) - len(stripped)
        m = re.match(r"^(\d+)[.)]\s+(.*)", stripped)
        if m:
            content = m.group(2).strip()
            level = 1 if indent >= 2 else 0
        elif re.match(r"^[-*+]\s", stripped):
            content = stripped[2:].strip()
            level = 1 if indent >= 2 else 0
        if content:
            content = content.replace("`", "").replace("**", "").strip()
            if content:
                status = "in_progress" if len(todos) == 0 else "pending"
                todos.append({"id": str(len(todos) + 1), "content": content, "status": status, "level": level})
                if len(todos) >= 20:
                    break
    return todos


# =============================================================================
# 3. Permission Manager
# =============================================================================


class PermissionManager:
    """Manages file-level write permissions with interactive approval."""

    def __init__(self, storage_file: str = ".permissions.json"):
        self.storage_file = Path(storage_file)
        self.data = self._load()

    def _load(self) -> dict:
        if self.storage_file.exists():
            with open(self.storage_file, "r") as f:
                return json.load(f)
        return {"granted": {}}

    def _save(self) -> None:
        with open(self.storage_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def check_permission(self, file_path: str) -> Optional[str]:
        path = Path(file_path).resolve()
        if str(path) in self.data["granted"]:
            return self.data["granted"][str(path)]
        for parent in path.parents:
            if str(parent) in self.data["granted"]:
                return self.data["granted"][str(parent)]
        return None

    def grant(self, path: str, scope: str) -> None:
        resolved = Path(path).resolve()
        if scope == "dir":
            p = resolved if resolved.is_dir() else resolved.parent
            self.data["granted"][str(p)] = "all_in_dir"
        else:
            self.data["granted"][str(resolved)] = "file"
        self._save()

    def check_and_request_permission(self, controller, file_path: str) -> bool:
        if self.check_permission(file_path):
            return True
        path_obj = Path(file_path).resolve()
        question = f"Need permission to access:\n" f"  File: {path_obj}\n" f"  Directory: {path_obj.parent}\n" f"What's your decision?"
        options = [
            f"Grant file access ({path_obj.name})",
            f"Grant directory access ({path_obj.parent.name})",
            "Deny",
        ]
        ask_tool = controller.registry.get("ask")
        if not ask_tool:
            return False
        choice = ask_tool.execute(controller.context, {"question": question, "options": options})
        if choice == "1":
            self.grant(file_path, "file")
            return True
        elif choice == "2":
            self.grant(file_path, "dir")
            return True
        return False


# =============================================================================
# 4. Tool Registry
# =============================================================================


class Registry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        # When True, writer tools are blocked (plan-mode gate).
        # Mirrors Harness executor.SetPlanMode.
        self.plan_mode: bool = False

    def add(self, tool: Tool) -> None:
        self._tools[tool.name()] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    def schemas(self) -> List[dict]:
        """Return tool schemas. In plan mode, omit writer tools so the model
        never sees them in its call list (mirrors Harness executor gate)."""
        if self.plan_mode:
            return [t.to_dict() for t in self._tools.values() if t.read_only()]
        return [t.to_dict() for t in self._tools.values()]

    def execute_gated(self, name: str, ctx: Any, args: dict) -> str:
        """Execute a tool, enforcing the plan-mode gate for writers."""
        tool = self.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found"

        # Plan-mode gate: writers are blocked unless specifically granted
        if self.plan_mode and not tool.read_only():
            # Check if this tool execution was already approved in plan mode
            # We use context to store approved writes during plan mode
            approved_writes = getattr(ctx, "approved_writes", set())
            write_key = f"{name}:{args.get('path', 'unknown')}"

            if write_key not in approved_writes:
                # Ask user for permission
                question = f"Plan mode is ON. Tool '{name}' is a writer (target: {args.get('path', 'unknown')}).\n" f"Allow this write operation?"
                options = ["Yes", "No (queue for later)"]

                ask_tool = self.get("ask")
                if ask_tool:
                    choice = ask_tool.execute(ctx, {"question": question, "options": options})
                    if choice == "1":
                        approved_writes.add(write_key)
                        setattr(ctx, "approved_writes", approved_writes)
                        return tool.execute(ctx, args)
                    else:
                        # Queue for later
                        pending = getattr(ctx, "pending_writes", [])
                        pending.append((tool, args))
                        setattr(ctx, "pending_writes", pending)
                        return f"[plan-mode] Tool '{name}' write blocked and queued for after plan approval."
                else:
                    return f"[plan-mode] Writer blocked (no ask tool available)."

        # Validate required parameters before execution.
        schema = tool.schema()
        required = schema.get("required", [])
        missing = [p for p in required if p not in args]
        if missing:
            return f"Error: tool '{name}' missing required parameters: {', '.join(missing)}"
        return tool.execute(ctx, args)


ALL_TOOLS = [ReadFileTool, WriteFileTool, EditFileTool, MultiEditTool, BashTool, GrepTool, GlobTool, LsTool, WebFetchTool, AskTool, TodoWriteTool]


def register_all_builtins(reg: Registry, cfg: Config, root: str, proxy=None) -> None:
    enabled = cfg.tools.enabled
    for cls in ALL_TOOLS:
        name = cls().name()
        if not should_enable(name, enabled):
            continue
        if cls is BashTool:
            reg.add(cls(prefer=cfg.tools.shell.prefer, path=cfg.tools.shell.path, timeout=cfg.tools.bash_timeout_seconds))
        elif cls is GrepTool:
            reg.add(cls(rg_path=""))
        elif cls is WebFetchTool:
            reg.add(cls(proxy=proxy))
        else:
            reg.add(cls())


# =============================================================================
# 4. Context / Message History
# =============================================================================


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message:
    def __init__(self, role: MessageRole, content: str, name: str = None, tool_call_id: str = None, tool_calls: list = None):
        self.role, self.content, self.name, self.tool_call_id, self.tool_calls = role, content, name, tool_call_id, tool_calls

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=MessageRole(d["role"]),
            content=d["content"],
            name=d.get("name"),
            tool_call_id=d.get("tool_call_id"),
            tool_calls=d.get("tool_calls"),
        )


class Context:
    def __init__(self, system_prompt: str, cfg: AgentConfig):
        self.system_prompt = system_prompt
        self.cfg = cfg
        self.messages: List[Message] = []
        self.todos: List[dict] = []

    def add_user(self, content: str) -> None:
        self.messages.append(Message(MessageRole.USER, content))

    def add_assistant(self, content: str, tool_calls: list = None) -> None:
        self.messages.append(Message(MessageRole.ASSISTANT, content, tool_calls=tool_calls))

    def add_tool_result(self, name: str, tid: str, result: str) -> None:
        self.messages.append(Message(MessageRole.TOOL, result, name=name, tool_call_id=tid))

    def estimate_tokens(self) -> int:
        total = len(self.system_prompt) + sum(len(m.content) for m in self.messages)
        return total // 4

    def compact(self, force: bool = False) -> None:
        pass

    def compact(self, max_tokens: int, force: bool = False) -> None:
        ratio = self.cfg.compact_force_ratio if force else self.cfg.compact_ratio
        effective_limit = max_tokens * ratio
        if self.estimate_tokens() < effective_limit:
            return
        # Simple compaction: summarize oldest messages
        to_compress = []
        keep = []
        for m in self.messages:
            if m.role == MessageRole.SYSTEM:
                continue
            if len(to_compress) < len(self.messages) // 2:
                to_compress.append(m)
            else:
                keep.append(m)
        if to_compress:
            summary = f"[Summary of {len(to_compress)} messages]"
            self.messages = [Message(MessageRole.ASSISTANT, summary)] + keep

    def to_openai(self) -> List[dict]:
        out = [{"role": "system", "content": self.system_prompt}]
        for m in self.messages:
            if m.role == MessageRole.SYSTEM:
                continue
            entry = {"role": m.role.value, "content": m.content}
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            out.append(entry)
        return out

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages],
            "todos": self.todos,
        }

    @classmethod
    def from_dict(cls, d: dict, cfg: AgentConfig) -> "Context":
        ctx = cls(system_prompt=d["system_prompt"], cfg=cfg)
        ctx.messages = [Message.from_dict(m) for m in d.get("messages", [])]
        ctx.todos = d.get("todos", [])
        return ctx


# =============================================================================
# 5. Provider / LLM Interface
# =============================================================================


class Provider:
    def __init__(self, entry: ProviderEntry):
        self.entry = entry
        self.api_key = os.environ.get(entry.api_key_env, "") or entry.api_key_env

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type((Exception,)), before_sleep=lambda retry_state: logger.warning(f"Retrying LLM call: attempt {retry_state.attempt_number}..."))
    def chat(self, messages: List[dict], tools: List[dict], temperature: float = 0.0) -> dict:
        """Send request to LLM using litellm (supporting OpenAI/Gemini/Anthropic, etc.)."""
        from litellm import completion, exceptions
        import time

        # Apply delay if configured
        if self.entry.delay_seconds > 0:
            logger.info(f"Delaying {self.entry.delay_seconds}s before request...")
            time.sleep(self.entry.delay_seconds)

        # Define retry capture logic
        def is_retryable(ex):
            return isinstance(
                ex,
                (
                    exceptions.RateLimitError,
                    exceptions.ServiceUnavailableError,
                    exceptions.APIError,
                    exceptions.Timeout,
                ),
            )

        kwargs = {
            "model": self.entry.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.entry.request_timeout,
            "api_key": self.api_key,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.entry.base_url:
            kwargs["api_base"] = self.entry.base_url

        try:
            response = completion(**kwargs)
            choice = response.choices[0]
            msg = choice.message
            tool_calls = []
            if msg.tool_calls:
                tool_calls = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
            return {"content": msg.content or "", "tool_calls": tool_calls, "finish_reason": choice.finish_reason or ""}
        except exceptions.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            return {"content": "Error: API Rate Limit Exceeded. Please wait a moment and try again.", "tool_calls": [], "finish_reason": "error"}
        except exceptions.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return {"content": "Error: Authentication failed. Check your API key.", "tool_calls": [], "finish_reason": "error"}
        except exceptions.ServiceUnavailableError as e:
            logger.error(f"Service Unavailable: {e}")
            return {"content": "Error: Service is currently unavailable (e.g. high load). Please try again in a few moments.", "tool_calls": [], "finish_reason": "error"}
        except Exception as e:
            if is_retryable(e):
                raise  # Trigger tenacity retry
            logger.error(f"LLM call error: {e}")
            return {"content": f"Error: {e}", "tool_calls": [], "finish_reason": "error"}


# =============================================================================
# 6. Agent Controller
# =============================================================================


class Controller:
    def __init__(self, workspace_root: str = "."):
        self.root = os.path.abspath(workspace_root)
        self.cfg = Config.load_for_root(self.root)
        self.registry = Registry()
        self.perm_manager = PermissionManager()
        self.provider: Optional[Provider] = None
        self.context: Optional[Context] = None
        self.step_count = 0
        # Plan-mode state: when True, PlanModeMarker is prepended to outgoing
        # turns and writer tools are blocked via registry.plan_mode.
        # Mirrors Harness Controller.planMode / SetPlanMode.
        self._plan_mode: bool = False

    def set_plan_mode(self, on: bool) -> None:
        """Toggle plan mode and synchronise the registry gate."""
        self._plan_mode = on
        self.registry.plan_mode = on

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def _sessions_dir(self) -> Path:
        d = Path.home() / ".reasonix" / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_session(self) -> str:
        import time, hashlib

        ts = time.strftime("%Y%m%d_%H%M%S")
        rand = hashlib.sha256(str(time.time()).encode()).hexdigest()[:6]
        sid = f"{ts}_{rand}"
        data = {
            "session_id": sid,
            "workspace_root": self.root,
            "step_count": self.step_count,
            "context": self.context.to_dict() if self.context else {},
        }
        path = self._sessions_dir() / f"{sid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return sid

    def load_session(self, sid: str) -> bool:
        path = self._sessions_dir() / f"{sid}.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.root = data.get("workspace_root", self.root)
        self.step_count = data.get("step_count", 0)
        ctx_data = data.get("context", {})
        if ctx_data:
            self.context = Context.from_dict(ctx_data, self.cfg.agent)
        return True

    def list_providers(self) -> List[str]:
        current_name = self.provider.entry.name if self.provider else None
        lines = []
        for i, p in enumerate(self.cfg.providers):
            active = " ◀ active" if p.name == current_name else ""
            lines.append(f"{i+1}. {p.name} ({p.model}){active}")
        return lines

    def switch_provider(self, name_or_idx: str) -> str:
        target = None
        if name_or_idx.isdigit():
            idx = int(name_or_idx) - 1
            if 0 <= idx < len(self.cfg.providers):
                target = self.cfg.providers[idx]
        else:
            for p in self.cfg.providers:
                if p.name == name_or_idx or p.model == name_or_idx:
                    target = p
                    break
        if not target:
            return f"Provider '{name_or_idx}' not found. Available:\n" + "\n".join(self.list_providers())
        self.provider = Provider(target)
        return f"Switched to provider: {target.name} ({target.model})"

    def reset_context(self) -> None:
        system_prompt = self.cfg.resolve_system_prompt(self.root)
        self.context = Context(system_prompt, self.cfg.agent)
        self.step_count = 0

    def boot(self) -> None:
        register_all_builtins(self.registry, self.cfg, self.root)
        if not self.cfg.providers:
            raise RuntimeError("No providers configured. Add at least one provider to config.yaml.")
        default = next((p for p in self.cfg.providers if p.default), self.cfg.providers[0])
        self.provider = Provider(default)
        system_prompt = self.cfg.resolve_system_prompt(self.root)
        self.context = Context(system_prompt, self.cfg.agent)

        _cmd = " ".join(getattr(sys, "orig_argv", sys.argv))
        self.context.add_user(f"System initialized. Service started with command: `{_cmd}`")

        log_box("boot", f"Workspace: {self.root}\nTools: {[t.name() for t in self.registry.list()]}\nSkills: {[s.name for s in self.cfg.enabled_skills()]}\nProvider: {self.provider.entry.name}\nModel: {self.provider.entry.model}\nBase URL: {self.provider.entry.base_url}")

    # ------------------------------------------------------------------
    # Plan-mode helpers
    # ------------------------------------------------------------------

    def _compose(self, text: str) -> str:
        """Prepend PlanModeMarker when plan mode is active.

        Mirrors Harness control.Controller.Compose: the marker rides the user
        message so the cache-stable system prefix is never modified.
        """
        if self._plan_mode:
            return self.cfg.plan_mode_marker + "\n\n" + text
        return text

    def _request_plan_approval(self, proposal: str) -> bool:
        """Show the plan proposal and ask the user to approve or reject.

        Returns True on approval. Mirrors Harness requestApproval called with
        planApprovalTool after a plan-mode turn finishes.
        """
        _stdout("\n" + "\u2550" * 60)
        _stdout("\U0001f4cb  PLAN MODE \u2014 proposed plan:")
        _stdout("\u2550" * 60)
        _stdout(proposal)
        _stdout("\u2550" * 60)
        if not sys.stdin.isatty():
            # Non-interactive: auto-approve (mirrors headless bot behaviour).
            _stdout("[non-interactive] Auto-approving plan.")
            return True
        while True:
            try:
                ans = input("Approve this plan? [y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _stdout("")
                return False
            if ans in ("y", "yes", ""):
                return True
            if ans in ("n", "no"):
                _stdout("Plan rejected. Plan mode remains active. Enter revised instructions.")
                return False
            _stdout("Please answer y (approve) or n (reject).")

    def _run_turn(self, composed_input: str) -> str:
        """Run one model turn (tool loop) and return the last assistant text.

        Separated from run() so the plan approval flow can call it for the
        follow-up execution turn without re-applying _compose. Uses
        registry.execute_gated so the plan-mode gate is enforced on every
        tool call inside this turn too.
        """
        self.context.add_user(composed_input)
        max_steps = self.cfg.agent.max_steps or 25
        last_content = ""
        try:
            for _ in range(max_steps):
                self.step_count += 1
                logger.info(f"--- Step {self.step_count} ---")
                # Get max tokens for current provider
                assert self.provider is not None, "Controller.provider must be initialized before running a turn"
                self.context.compact(max_tokens=self.provider.entry.context_window, force=False)
                messages = self.context.to_openai()
                tools = self.registry.schemas()
                if not self.provider:
                    return "Error: no provider"
                # KeyboardInterrupt propagates upward to be handled by the CLI layer.
                try:
                    response = self.provider.chat(messages, tools, self.cfg.agent.temperature)
                except Exception as e:
                    sys.stdout.write(f"\n⚠️  LLM call failed after retries: {e}\n")
                    sys.stdout.flush()
                    return f"Error: LLM call failed — {e}"
                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])
                finish = response.get("finish_reason", "")
                model_parts = []
                if finish:
                    model_parts.append(f"[finish={finish}]")
                if content:
                    model_parts.append(content[:500])
                if tool_calls:
                    tc_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    model_parts.append(f"\u2192 tools: {', '.join(tc_names)}")
                if model_parts:
                    log_box("model", "\n".join(model_parts))
                self.context.add_assistant(content or "", tool_calls=tool_calls or None)
                last_content = content or last_content
                if tool_calls:
                    for tc in tool_calls:
                        tid = tc.get("id", "unknown")
                        fn = tc.get("function", {})
                        tname = fn.get("name", "")
                        try:
                            args = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
                        except json.JSONDecodeError:
                            args = {}
                        # execute_gated blocks writers when plan_mode is on.
                        result = self.registry.execute_gated(tname, self, args)
                        self.context.add_tool_result(tname, tid, result)

                        def format_args(args_dict):
                            try:
                                # Recursively truncate strings in dictionary
                                def truncate(v):
                                    if isinstance(v, str) and len(v) > 50:
                                        return v[:47] + "..."
                                    if isinstance(v, dict):
                                        return {k: truncate(val) for k, val in v.items()}
                                    if isinstance(v, list):
                                        return [truncate(val) for val in v]
                                    return v

                                truncated = truncate(args_dict)
                                s = json.dumps(truncated, ensure_ascii=False)
                                return s[:200] + "..." if len(s) > 200 else s
                            except:
                                return str(args_dict)[:200]

                        call_str = f"call: {tname}\nargs: {format_args(args)}"
                        res_str = f"result:\n{result[:600]}"
                        log_box("tool", f"{call_str}\n{'-' * 36}\n{res_str}")
                else:
                    return content or "(no response)"
                if finish == "stop":
                    return content or "(stopped)"
        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user. Context retained.")
            return "(Interrupted by user)"
        return last_content or "(max_steps reached)"

    def run(self, user_request: str) -> str:
        """Run a user request, honouring plan mode.

        Plan-mode flow (mirrors Harness runTurnWithRawDisplay):
          1. Prepend PlanModeMarker and run a read-only research/planning turn.
          2. Present the proposal to the user for approval.
          3a. Approved  -> exit plan mode, seed todos, run execution turn.
          3b. Rejected  -> stay in plan mode; user can revise and re-submit.

        Normal flow: just run the tool loop.
        """
        if self.context is None:
            self.boot()

        log_box("user", user_request[:500])
        composed = self._compose(user_request)

        if not self._plan_mode:
            # Normal (non-plan) execution path.
            return self._run_turn(composed)

        # ── Plan mode: research / planning turn ───────────────────────────────
        proposal = self._run_turn(composed)

        if not proposal or not proposal.strip():
            return "(plan mode: no proposal generated)"

        # ── Approval gate ─────────────────────────────────────────────────────
        approved = self._request_plan_approval(proposal)

        if not approved:
            # Keep plan mode on so the user can refine and re-submit.
            return "Plan rejected. Plan mode is still active. Send revised instructions."

        # ── Approved: exit plan mode and execute ──────────────────────────────
        _stdout("\n\u2705 Plan approved \u2014 executing...")

        # Execute any pending writes queued during plan mode
        pending = getattr(self.context, "pending_writes", [])
        if pending:
            _stdout(f"\n\u2699 Executing {len(pending)} queued write operations...")
            for tool, args in pending:
                _stdout(f"  Running {tool.name()} on {args.get('path', 'unknown')}...")
                tool.execute(self, args)

        self.set_plan_mode(False)

        # Seed a starter todo list from the plan (mirrors seedPlanTodos).
        todos = parse_plan_todos(proposal)
        if todos and self.context is not None:
            self.context.todos = todos
            todo_log = "\n".join(f"  [{t['status']}] {t['content']}" for t in todos)
            log_box("tool", f"todo_write (plan seed)\n{todo_log}")

        # Execution turn with plan-approved nudge (mirrors planApprovedMessage turn).
        return self._run_turn(self.cfg.plan_approved_message)


# =============================================================================
# 7. CLI Entry Point
# =============================================================================


def _read_input_auto(timeout: float = 0.08) -> str:
    """Read user input using prompt_toolkit. Supports multiline via Alt+Enter."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings

    if not hasattr(_read_input_auto, "_session"):
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        _read_input_auto._session = PromptSession(key_bindings=bindings, multiline=False)  # Enter submits; Alt+Enter for newline

    if not sys.stdin.isatty():
        # Non-interactive fallback
        line = sys.stdin.readline()
        if not line:
            raise EOFError()
        return line.rstrip("\n")

    session = _read_input_auto._session
    result = session.prompt("▶ ")
    return result


COMMANDS_HELP = """
Harness Command Help
────────────────────────────────────────────────────────────────────────────────
/new                   — Start a new conversation (automatically saves the current session)
/clear                 — Same as /new, clears the current conversation context
/model                 — List all available providers
/model <name or index> — Switch to the specified provider (supports name or list index)
/plan                  — Enable plan mode (next request will be planned first, then await confirmation)
/plan on               — Enable plan mode
/plan off              — Disable plan mode (unlocks write operations)
/plan status           — Check current plan mode status
Ctrl-C                 — Cancel and exit
Ctrl-D / EOF           — Exit (automatically saves session)
/exit, /quit           — Exit (automatically saves session)
────────────────────────────────────────────────────────────────────────────────
"""


def main(argv=None) -> None:
    import argparse

    _load_dotenv()

    parser = argparse.ArgumentParser(description="Harness Kernel (Python)")
    parser.add_argument("--root", default=".", help="Workspace root")
    parser.add_argument("--model", default="", help="Override default model")
    parser.add_argument("--resume", default="", help="Resume session ID")
    parser.add_argument("request", nargs="?", help="User request to execute")
    args = parser.parse_args(argv)
    ctrl = Controller(workspace_root=args.root)

    if args.resume:
        if ctrl.load_session(args.resume):
            register_all_builtins(ctrl.registry, ctrl.cfg, ctrl.root)
            if not ctrl.cfg.providers:
                raise RuntimeError("No providers configured. Add at least one provider to config.yaml.")
            default = next((p for p in ctrl.cfg.providers if p.default), ctrl.cfg.providers[0])
            ctrl.provider = Provider(default)
            log_box("boot", f"Resumed session: {args.resume}\nWorkspace: {ctrl.root}\nMessages: {len(ctrl.context.messages)}")
        else:
            _stdout(f"Session '{args.resume}' not found. Starting fresh.")
            ctrl.boot()
    else:
        ctrl.boot()

    if args.request:
        _stdout(f"\n=== Result ===")
        rich_print(ctrl.run(args.request))
    else:
        _stdout(COMMANDS_HELP)
        while True:
            try:
                req = _read_input_auto()
                if not req:
                    continue
            except (EOFError, KeyboardInterrupt):
                _stdout("")
                break
            req = req.strip()
            if not req:
                continue
            if req in ("/help", "?"):
                _stdout(COMMANDS_HELP)
                continue
            if req in ("/exit", "/quit"):
                break
            if req in ("/new", "/clear"):
                sid = ctrl.save_session()
                ctrl.reset_context()
                _stdout(f"New context started. Previous session: --resume {sid}")
                continue
            if req.startswith("/model"):
                parts = req.split(None, 1)
                if len(parts) == 1:
                    _stdout("\n".join(ctrl.list_providers()))
                else:
                    _stdout(ctrl.switch_provider(parts[1]))
                continue
            if req == "/context":
                if ctrl.context:
                    # 获取即将发送给 LLM 的数据结构
                    messages = ctrl.context.to_openai()
                    tools = ctrl.registry.schemas()
                    payload = {"model": ctrl.provider.entry.model if ctrl.provider else "default", "messages": messages, "tools": tools, "temperature": ctrl.cfg.agent.temperature, "todos": ctrl.context.todos}
                    _stdout("\n--- Simulated LLM Request Payload ---")
                    _stdout(json.dumps(payload, indent=2, ensure_ascii=False))
                    _stdout("-------------------------------------\n")
                else:
                    _stdout("Context is empty.")
                continue
            if req == "/skills":
                _stdout(f"Available skills:\n" + "\n".join([f"/{s.name} — {s.description[:47] + '...' if len(s.description) > 50 else s.description}" for s in ctrl.cfg.enabled_skills()]))
                continue
            if req.startswith("/") and ctrl.cfg.get_skill(req[1:].split()[0]):
                skill_name, *skill_args = req[1:].split()
                skill = ctrl.cfg.get_skill(skill_name)
                _stdout(f"Triggering skill: {skill_name} with args: {skill_args}")
                # 实际执行逻辑：将 skill.body 和参数注入到 context 中进行对话
                ctrl.context.add_user(f"Execute skill {skill_name} with args: {' '.join(skill_args)}\n\nSkill directory: {skill.path}\n\nSkill definition:\n{skill.body}")
                _stdout("")
                rich_print(ctrl.run("Proceed with this skill execution"))
                _stdout("")
                continue
            if req == "/plan" or req.startswith("/plan "):
                parts = req.split(None, 1)
                sub = parts[1].strip().lower() if len(parts) > 1 else "on"
                if sub in ("off", "disable", "false", "0"):
                    ctrl.set_plan_mode(False)
                    _stdout("Plan mode: OFF — writers unblocked.")
                elif sub in ("status",):
                    state = "ON" if ctrl.plan_mode else "OFF"
                    _stdout(f"Plan mode: {state}")
                else:
                    # "on" / "enable" / bare "/plan"
                    ctrl.set_plan_mode(True)
                    _stdout("Plan mode: ON — next request will be planned before execution.")
                    _stdout("  Writers are blocked until you approve the plan.")
                    _stdout("  Use /plan off to cancel without sending a request.")
                continue
            try:
                _stdout("")
                rich_print(ctrl.run(req))
                _stdout("")
            except KeyboardInterrupt:
                _stdout("\n\n⚠️  Cancelled (Ctrl+C). Exiting.")
                break
        sid = ctrl.save_session()
        _stdout(f"\nSession saved. Resume with: --resume {sid}")


# Rebuild models to resolve forward references
Config.model_rebuild()
AgentConfig.model_rebuild()
ToolsConfig.model_rebuild()
SkillsConfig.model_rebuild()
PermissionsConfig.model_rebuild()
SandboxConfig.model_rebuild()
ShellConfig.model_rebuild()
ProviderEntry.model_rebuild()
SkillEntry.model_rebuild()

if __name__ == "__main__":
    import sys

    _load_dotenv()

    if "ipykernel" not in sys.argv[0]:
        main(sys.argv[1:])
