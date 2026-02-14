# Sessions

Agent sessions provide isolated workspaces with their own logs and memory.

## Layout

```
sessions/<name>/
  logs/          # 001_agent_<ts>.jsonl, 001_agent_<ts>.log, ...
  workspace/     # Agent's sandboxed filesystem root.
  memory.db      # Session-specific semantic memory.
```

## Modes

- **Default** (no flags): shared `logs/` + `memory/memory.db`, no workspace.
- **Named** (`--session NAME`): own workspace, own memory, own logs.
- **New** (`--new-session`): like named, with auto-generated UUID name.
- **Resume** (`--session NAME --resume`): replay previous log, continue in same workspace.

## Instance tracking

Each run in a session increments the instance counter (derived from JSONL file count in logs dir). The instance number is:
- Prefixed on log filenames (`001_agent_*.jsonl`).
- Injected into the system prompt ("You are instance N.").
- Reported by the status tool.

## Workspace templates

`workspace_template/` contents are copied into fresh session workspaces. Skipped if workspace already exists (resume case).

## Context transfer between instances

| Mechanism | What persists | Cost to next instance |
|-----------|---------------|-----------------------|
| Session recap | Procedural context (what was I doing, unfinished work) | Zero (injected into system prompt) |
| Semantic memory | Declarative knowledge (facts, conclusions) | Recall queries |
| Workspace files | Artifacts the agent created | ls + read |
| JSONL resume | Full conversation replay | Expensive (full history) |
| System prompt | Instance number | Zero |

### Session recap

At shutdown, the agent generates a recap from the current JSONL log and writes it to `workspace/_recap.md`. The next instance loads this file and appends it to the system prompt. See `docs/design/session_recap.md` for design details.
