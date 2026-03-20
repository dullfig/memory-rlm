# ClaudeRLM

Persistent project memory for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). A Rust MCP server that transparently indexes your conversation context and code changes, then re-injects the most relevant information when context is lost to compaction.

## Why this exists

Claude Code sessions have finite context windows. As conversations grow through requests, code edits, debugging, and planning, older context gets compressed or discarded. You lose decisions, rationale, and the thread of what you were doing.

ClaudeRLM fixes this by indexing everything as it happens and surgically re-injecting what matters when context is lost.

## How it works

The core idea is borrowed from [Recursive LLMs](https://arxiv.org/abs/2512.24601) (Zhang et al., 2025), which showed that LLMs can process arbitrarily long inputs by treating them as external data to be programmatically queried rather than consumed all at once. Our insight: conversation context grows incrementally, so it can be indexed at write-time as each turn happens, rather than processed at read-time when it's already too late. This makes indexing nearly free -- each hook call takes milliseconds -- and retrieval is a fast SQLite query.

In practice, Claude Code [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) fire on every user prompt, code edit, file read, and bash command. ClaudeRLM captures each event into a local SQLite database with FTS5 full-text search. When compaction is about to happen, a `PreCompact` hook ensures everything is indexed and creates a checkpoint summary. After compaction, a `SessionStart` hook queries the index and injects the most relevant context back into the conversation -- ranked by recency, type importance, and file affinity -- so Claude picks up right where it left off.

## Features

- **Passive indexing** -- hooks fire automatically, Claude never needs to decide to use it
- **Full-text search** over conversation history (SQLite FTS5 with BM25 ranking)
- **Code structure indexing** via tree-sitter (Rust, Python, TypeScript, JavaScript, Go, C, C++)
- **Background file watcher** for incremental re-indexing on file changes
- **Ranked context injection** after compaction (type weight x recency x file affinity)
- **Knowledge distillation** at session end -- extracts decisions, preferences, conventions, and bug fixes
- **LLM-enhanced distillation** with Haiku (~1 cent/session) or any OpenAI-compatible endpoint (Ollama for free)
- **4 MCP tools** for explicit search when needed: `memory_search`, `memory_symbols`, `memory_decisions`, `memory_files`
- **Cross-session memory** -- knowledge persists and is injected at the start of every new session

## Install

### Plugin install (recommended)

Install as a Claude Code plugin for one-command setup with automatic binary management:

```
/plugin marketplace add dullfig/claude-plugins
/plugin install memory-rlm
```

The binary is downloaded automatically on first session start. No PATH configuration needed.

To update, delete the cached binary and restart:

```bash
rm -f <plugin-dir>/bin/memory-rlm    # Unix
del <plugin-dir>\bin\memory-rlm.exe  # Windows
```

### Manual install

If you prefer not to use the plugin system, standalone installers are still available:

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/dullfig/memory-rlm/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/dullfig/memory-rlm/main/install.ps1 | iex
```

**From source (any platform with Rust):**
```bash
cargo install --git https://github.com/dullfig/memory-rlm.git
```

If you install manually, you'll need to configure hooks yourself (see Manual hook setup below). Use **absolute paths** to the binary — Claude Code spawns hook subprocesses without inheriting your shell PATH.

### Migrating from manual install to plugin

1. Install the plugin: `/plugin marketplace add dullfig/claude-plugins` then `/plugin install memory-rlm`
2. Remove old hooks from `~/.claude/settings.json` (any entries referencing `memory-rlm`)
3. Optionally remove the old binary from `~/.local/bin/memory-rlm` or `%LOCALAPPDATA%\Programs\memory-rlm\`

Your existing database (`.claude/memory-rlm.db`) is preserved — all your indexed memory carries over.

## Manual hook setup

If you installed manually, copy `hooks.example.json` into your project's `.claude/settings.json` (or merge into your existing settings).

**Important:** Use the absolute path to `memory-rlm` in hook commands. Replace `/path/to/memory-rlm` with the actual install location (e.g., `~/.local/bin/memory-rlm` on Linux/macOS, `C:/Users/YOU/AppData/Local/Programs/memory-rlm/memory-rlm.exe` on Windows):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [{ "type": "command", "command": "/path/to/memory-rlm index-prompt", "timeout": 5 }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "/path/to/memory-rlm index-edit", "timeout": 5 }]
      },
      {
        "matcher": "Read",
        "hooks": [{ "type": "command", "command": "/path/to/memory-rlm index-read", "timeout": 2 }]
      },
      {
        "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "/path/to/memory-rlm index-bash", "timeout": 2 }]
      }
    ],
    "PreCompact": [
      {
        "hooks": [{ "type": "command", "command": "/path/to/memory-rlm pre-compact", "timeout": 10 }]
      }
    ],
    "SessionStart": [
      {
        "hooks": [{ "type": "command", "command": "/path/to/memory-rlm session-start", "timeout": 10 }]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [{ "type": "command", "command": "/path/to/memory-rlm session-end", "timeout": 30 }]
      }
    ]
  }
}
```

## MCP server (optional)

Add to your Claude Code MCP settings for explicit search tools:

```json
{
  "mcpServers": {
    "memory-rlm": {
      "command": "memory-rlm",
      "args": ["serve"]
    }
  }
}
```

Plugin users get this automatically via the plugin's `.mcp.json`.

## LLM distillation (optional)

For higher-quality knowledge extraction at session end, add your API key to a config file.

**Project-level** (`.claude/memory-rlm.toml` in your project):
```toml
[llm]
provider = "anthropic"
api_key = "sk-ant-..."
```

**Global** (`~/.config/memory-rlm/config.toml` on Linux/macOS, `%APPDATA%\memory-rlm\config.toml` on Windows):
```toml
[llm]
provider = "anthropic"
api_key = "sk-ant-..."
model = "claude-haiku-4-5-20251001"   # default, cheapest option
```

**For a local model via Ollama (free):**
```toml
[llm]
provider = "ollama"
model = "llama3"
```

Project-level config takes priority over global. Environment variables (`CONTEXTMEM_LLM_*`) are also supported as a fallback.

Without any LLM configured, ClaudeRLM falls back to heuristic pattern matching for distillation, which still works well for common patterns (technology choices, "always/never" preferences, build tools, test frameworks).

| Config key | Env var fallback | Default | Description |
|---|---|---|---|
| `llm.api_key` | `CONTEXTMEM_LLM_API_KEY` | *(none)* | API key (required for cloud, optional for Ollama) |
| `llm.provider` | `CONTEXTMEM_LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `llm.model` | `CONTEXTMEM_LLM_MODEL` | `claude-haiku-4-5-20251001` | Model name |
| `llm.base_url` | `CONTEXTMEM_LLM_BASE_URL` | *(provider default)* | Custom endpoint URL |

## What gets indexed

| Event | What's captured |
|---|---|
| User prompt | Full request text |
| Edit/Write | File path, old/new content, change description |
| Read | File path |
| Bash | Command and output (truncated to 2KB) |
| PreCompact | Checkpoint summary of all activity so far |
| Session end | Session summary + distilled knowledge |

## What gets injected

**At session start:** project structure, recent session summaries, distilled knowledge (decisions, conventions, preferences).

**After compaction:** checkpoint summaries, all user requests from the session, active file list, then the highest-ranked remaining turns up to a 16K character budget.

## Disable / enable (kill switch)

If something goes wrong and ClaudeRLM is interfering with your session, disable it from any terminal:

```bash
memory-rlm disable        # All hooks silently exit, Claude Code works normally
memory-rlm enable         # Re-enable hooks
memory-rlm status         # Shows DISABLED/enabled state
```

This creates/removes a `~/.memory-rlm-disabled` flag file. No Claude session needed -- just open a terminal and type the command.

## CLI commands

```
memory-rlm serve          # Start MCP server (default)
memory-rlm status         # Show index statistics
memory-rlm --version      # Show version
memory-rlm disable        # Disable all hooks (emergency kill switch)
memory-rlm enable         # Re-enable hooks
memory-rlm index-prompt   # Hook: index user prompt (stdin)
memory-rlm index-edit     # Hook: index code edit (stdin)
memory-rlm index-read     # Hook: index file read (stdin)
memory-rlm index-bash     # Hook: index bash command (stdin)
memory-rlm pre-compact    # Hook: pre-compaction checkpoint
memory-rlm session-start  # Hook: inject context
memory-rlm session-end    # Hook: distill + summarize
```

## Data storage

All data is stored locally in `.claude/memory-rlm.db` (SQLite) inside your project directory. Nothing leaves your machine unless you configure LLM distillation with a cloud API.

## Supported languages (tree-sitter)

Rust, Python, TypeScript, TSX, JavaScript, Go, C, C++

## License

Business Source License 1.1 (BSL 1.1). Free for all use including production, except offering it as a competing commercial hosted service. Converts to Apache 2.0 on February 14, 2029. See [LICENSE](LICENSE) for details.
