#!/usr/bin/env bash
set -euo pipefail

# DEPRECATED: Use the Claude Code plugin instead:
#   /plugin marketplace add dullfig/claude-plugins
#   /plugin install memory-rlm
# This standalone installer will be removed in a future version.

# ClaudeRLM installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/dullfig/memory-rlm/main/install.sh | bash

REPO="dullfig/memory-rlm"
INSTALL_DIR="${HOME}/.local/bin"
BINARY="memory-rlm"

info()  { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m==>\033[0m %s\n" "$*" >&2; }

# Detect platform
detect_platform() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)  os="unknown-linux-gnu" ;;
        Darwin) os="apple-darwin" ;;
        *)      error "Unsupported OS: $os"; exit 1 ;;
    esac

    case "$arch" in
        x86_64|amd64)  arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *)             error "Unsupported architecture: $arch"; exit 1 ;;
    esac

    # Only macOS has aarch64 builds; Linux aarch64 falls back to cargo install
    if [ "$arch" = "aarch64" ] && [ "$os" = "unknown-linux-gnu" ]; then
        echo ""
        return
    fi

    echo "${arch}-${os}"
}

# Get latest release tag from GitHub
get_latest_version() {
    if command -v curl &>/dev/null; then
        curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//'
    elif command -v wget &>/dev/null; then
        wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//'
    else
        echo ""
    fi
}

# Download pre-built binary
download_binary() {
    local platform="$1" version="$2"
    local url="https://github.com/${REPO}/releases/download/${version}/memory-rlm-${platform}.tar.gz"
    local tmpdir
    tmpdir="$(mktemp -d)"

    info "Downloading ${BINARY} ${version} for ${platform}..."

    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "${tmpdir}/archive.tar.gz"
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "${tmpdir}/archive.tar.gz"
    else
        error "Neither curl nor wget found"
        return 1
    fi

    tar xzf "${tmpdir}/archive.tar.gz" -C "${tmpdir}"
    mkdir -p "$INSTALL_DIR"
    mv "${tmpdir}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
    chmod +x "${INSTALL_DIR}/${BINARY}"
    rm -rf "$tmpdir"
}

# Build from source using cargo
build_from_source() {
    if ! command -v cargo &>/dev/null; then
        error "No pre-built binary available for your platform and cargo is not installed."
        error "Install Rust from https://rustup.rs and try again."
        exit 1
    fi

    info "Building from source with cargo install..."
    cargo install --git "https://github.com/${REPO}.git"
}

# Configure Claude Code hooks
configure_hooks() {
    local settings_dir="${HOME}/.claude"
    local settings_file="${settings_dir}/settings.json"

    if [ ! -d "$settings_dir" ]; then
        info "Claude Code settings directory not found at ${settings_dir}"
        info "Skipping hook configuration. Run claude once first, then re-run this installer."
        return
    fi

    info "Configuring Claude Code hooks..."

    # Use absolute path — Claude Code's hook subprocess doesn't inherit the user's shell PATH
    local exe="${INSTALL_DIR}/${BINARY}"

    # Build the hooks JSON
    local hooks_json
    hooks_json=$(cat <<HOOKS_EOF
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [{ "type": "command", "command": "${exe} index-prompt", "timeout": 5 }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "${exe} index-edit", "timeout": 5 }]
      },
      {
        "matcher": "Read",
        "hooks": [{ "type": "command", "command": "${exe} index-read", "timeout": 2 }]
      },
      {
        "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "${exe} index-bash", "timeout": 2 }]
      }
    ],
    "PreCompact": [
      {
        "hooks": [{ "type": "command", "command": "${exe} pre-compact", "timeout": 10 }]
      }
    ],
    "SessionStart": [
      {
        "hooks": [{ "type": "command", "command": "${exe} session-start", "timeout": 10 }]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [{ "type": "command", "command": "${exe} session-end", "timeout": 30 }]
      }
    ]
  }
}
HOOKS_EOF
)

    if [ -f "$settings_file" ]; then
        # Check if hooks are already configured
        if grep -q "memory-rlm" "$settings_file" 2>/dev/null; then
            info "Hooks already configured in ${settings_file}"
            return
        fi

        # Merge hooks into existing settings using Python (widely available)
        if command -v python3 &>/dev/null; then
            python3 -c "
import json, sys
with open('$settings_file') as f:
    settings = json.load(f)
hooks = json.loads('''$hooks_json''')
settings.setdefault('hooks', {}).update(hooks['hooks'])
with open('$settings_file', 'w') as f:
    json.dump(settings, f, indent=2)
" && ok "Hooks merged into ${settings_file}" && return
        fi

        info "Existing settings.json found but can't auto-merge."
        info "Please manually add hooks from hooks.example.json to ${settings_file}"
    else
        echo "$hooks_json" > "$settings_file"
        ok "Created ${settings_file} with hooks configuration"
    fi
}

# Check PATH
check_path() {
    if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
        info "${INSTALL_DIR} is not in your PATH. Add it:"
        echo ""
        echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
        echo ""
    fi
}

# Sync plugin cache — copy installed binary into Claude Code's plugin cache
sync_plugin_cache() {
    local cache_base="${HOME}/.claude/plugins/cache/dullfig-plugins/memory-rlm"
    [ -d "$cache_base" ] || return 0

    local src="${INSTALL_DIR}/${BINARY}"
    [ -f "$src" ] || return 0

    for version_dir in "$cache_base"/*/; do
        local bin_dir="${version_dir}bin"
        if [ -d "$bin_dir" ]; then
            if cp "$src" "${bin_dir}/${BINARY}" 2>/dev/null; then
                chmod +x "${bin_dir}/${BINARY}"
                ok "Synced plugin cache: ${bin_dir}/${BINARY}"
            else
                info "Could not sync plugin cache at ${bin_dir}/${BINARY}"
            fi
        fi
    done
}

# Main
main() {
    echo ""
    echo "  ClaudeRLM Installer"
    echo "  Persistent project memory for Claude Code"
    echo ""

    local platform version

    platform="$(detect_platform)"
    version="$(get_latest_version)"

    if [ -n "$platform" ] && [ -n "$version" ]; then
        download_binary "$platform" "$version" || build_from_source
    else
        if [ -z "$version" ]; then
            info "No releases found, building from source..."
        elif [ -z "$platform" ]; then
            info "No pre-built binary for your platform, building from source..."
        fi
        build_from_source
    fi

    ok "Installed ${BINARY} to ${INSTALL_DIR}"
    echo ""

    configure_hooks
    sync_plugin_cache
    check_path

    echo ""
    ok "Done! Start a Claude Code session to see it in action."
    echo ""
    echo "  Optional: set CONTEXTMEM_LLM_API_KEY for LLM-enhanced distillation"
    echo "  See https://github.com/${REPO} for details"
    echo ""
}

main "$@"
