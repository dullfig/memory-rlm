# /memory-rlm:status

Show ClaudeRLM memory status and statistics for the current project.

Run:
```bash
"${CLAUDE_PLUGIN_ROOT}/bin/memory-rlm" status
```

This shows:
- Whether ClaudeRLM is enabled or disabled
- Number of sessions, turns, and knowledge entries indexed
- Symbol counts by kind (functions, structs, classes, etc.)
- Sample indexed symbols
- Distilled knowledge (decisions, conventions, preferences)
