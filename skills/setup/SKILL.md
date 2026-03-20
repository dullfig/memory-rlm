# ClaudeRLM Setup

Use this skill to set up ClaudeRLM when the bootstrap script hasn't run yet or if you need to manually install the binary.

## Steps

1. Check if the binary exists at `${CLAUDE_PLUGIN_ROOT}/bin/memory-rlm`:

```bash
ls -la "${CLAUDE_PLUGIN_ROOT}/bin/memory-rlm" 2>/dev/null || echo "Not found"
```

2. If the binary is missing, run the bootstrap script:

**Unix (Linux/macOS):**
```bash
"${CLAUDE_PLUGIN_ROOT}/scripts/bootstrap"
```

**Windows:**
```cmd
"%CLAUDE_PLUGIN_ROOT%\scripts\bootstrap.cmd"
```

3. If bootstrap fails (network issues, etc.), download manually:

- Go to https://github.com/dullfig/memory-rlm/releases/latest
- Download the archive for your platform:
  - Linux x86_64: `memory-rlm-x86_64-unknown-linux-gnu.tar.gz`
  - macOS x86_64: `memory-rlm-x86_64-apple-darwin.tar.gz`
  - macOS ARM: `memory-rlm-aarch64-apple-darwin.tar.gz`
  - Windows: `memory-rlm-x86_64-pc-windows-msvc.zip`
- Extract the binary to `${CLAUDE_PLUGIN_ROOT}/bin/`
- On Unix, make it executable: `chmod +x ${CLAUDE_PLUGIN_ROOT}/bin/memory-rlm`

4. Verify the installation:

```bash
"${CLAUDE_PLUGIN_ROOT}/bin/memory-rlm" --version
```

5. Check memory status:

```bash
"${CLAUDE_PLUGIN_ROOT}/bin/memory-rlm" status
```
