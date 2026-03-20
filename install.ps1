# DEPRECATED: Use the Claude Code plugin instead:
#   /plugin marketplace add dullfig/claude-plugins
#   /plugin install memory-rlm
# This standalone installer will be removed in a future version.

# ClaudeRLM installer for Windows
# Usage: irm https://raw.githubusercontent.com/dullfig/memory-rlm/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo = "dullfig/memory-rlm"
$Binary = "memory-rlm.exe"
$InstallDir = "$env:LOCALAPPDATA\Programs\memory-rlm"

function Write-Info($msg)  { Write-Host "==> $msg" -ForegroundColor Blue }
function Write-Ok($msg)    { Write-Host "==> $msg" -ForegroundColor Green }
function Write-Err($msg)   { Write-Host "==> $msg" -ForegroundColor Red }

# Get latest release tag
function Get-LatestVersion {
    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
        return $release.tag_name
    } catch {
        return $null
    }
}

# Download pre-built binary
function Install-Binary {
    param([string]$Version)

    $target = "x86_64-pc-windows-msvc"
    $url = "https://github.com/$Repo/releases/download/$Version/memory-rlm-$target.zip"
    $tmpDir = Join-Path $env:TEMP "memory-rlm-install"
    $zipPath = Join-Path $tmpDir "memory-rlm.zip"

    Write-Info "Downloading memory-rlm $Version for Windows..."

    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

    Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Move-Item -Path (Join-Path $tmpDir $Binary) -Destination (Join-Path $InstallDir $Binary) -Force

    Remove-Item -Path $tmpDir -Recurse -Force
}

# Build from source
function Install-FromSource {
    $cargo = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargo) {
        Write-Err "No pre-built binary available and cargo is not installed."
        Write-Err "Install Rust from https://rustup.rs and try again."
        exit 1
    }

    Write-Info "Building from source with cargo install..."
    cargo install --git "https://github.com/$Repo.git"
}

# Add to user PATH
function Add-ToPath {
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($userPath -notlike "*$InstallDir*") {
        Write-Info "Adding $InstallDir to user PATH..."
        [Environment]::SetEnvironmentVariable("PATH", "$InstallDir;$userPath", "User")
        $env:PATH = "$InstallDir;$env:PATH"
        Write-Ok "Added to PATH (restart your terminal for it to take effect)"
    }

    # Also add to .bashrc for Git Bash (used by Claude Code hooks on Windows)
    $bashrc = Join-Path $env:USERPROFILE ".bashrc"
    # Git Bash uses /c/ style paths, not C:/
    $bashPath = $InstallDir -replace '\\', '/'
    $driveLetter = $bashPath.Substring(0, 1).ToLower()
    $bashPath = "/$driveLetter$($bashPath.Substring(2))"
    $exportLine = "export PATH=`"$bashPath:`$PATH`""
    if ((Test-Path $bashrc) -and (Get-Content $bashrc -Raw) -match [regex]::Escape("memory-rlm")) {
        # Already in .bashrc
    } else {
        Write-Info "Adding $InstallDir to ~/.bashrc for Git Bash..."
        Add-Content -Path $bashrc -Value "`n# ClaudeRLM`n$exportLine"
        Write-Ok "Added to ~/.bashrc"
    }
}

# Configure Claude Code hooks
function Configure-Hooks {
    $settingsDir = Join-Path $env:USERPROFILE ".claude"
    $settingsFile = Join-Path $settingsDir "settings.json"

    if (-not (Test-Path $settingsDir)) {
        Write-Info "Claude Code settings directory not found at $settingsDir"
        Write-Info "Skipping hook configuration. Run claude once first, then re-run this installer."
        return
    }

    Write-Info "Configuring Claude Code hooks..."

    # Use absolute path — Claude Code's hook subprocess doesn't inherit the user's shell PATH
    $exe = "$InstallDir\memory-rlm.exe" -replace '\\', '/'

    $hooks = @{
        hooks = @{
            UserPromptSubmit = @(
                @{ hooks = @( @{ type = "command"; command = "$exe index-prompt"; timeout = 5 } ) }
            )
            PostToolUse = @(
                @{ matcher = "Edit|Write"; hooks = @( @{ type = "command"; command = "$exe index-edit"; timeout = 5 } ) }
                @{ matcher = "Read"; hooks = @( @{ type = "command"; command = "$exe index-read"; timeout = 2 } ) }
                @{ matcher = "Bash"; hooks = @( @{ type = "command"; command = "$exe index-bash"; timeout = 2 } ) }
            )
            PreCompact = @(
                @{ hooks = @( @{ type = "command"; command = "$exe pre-compact"; timeout = 10 } ) }
            )
            SessionStart = @(
                @{ hooks = @( @{ type = "command"; command = "$exe session-start"; timeout = 10 } ) }
            )
            SessionEnd = @(
                @{ hooks = @( @{ type = "command"; command = "$exe session-end"; timeout = 30 } ) }
            )
        }
    }

    if (Test-Path $settingsFile) {
        $content = Get-Content $settingsFile -Raw
        if ($content -match "memory-rlm") {
            Write-Info "Hooks already configured in $settingsFile"
            return
        }

        try {
            $settings = $content | ConvertFrom-Json -AsHashtable
            if (-not $settings.ContainsKey("hooks")) {
                $settings["hooks"] = @{}
            }
            foreach ($key in $hooks.hooks.Keys) {
                $settings["hooks"][$key] = $hooks.hooks[$key]
            }
            $settings | ConvertTo-Json -Depth 10 | Set-Content $settingsFile -Encoding UTF8
            Write-Ok "Hooks merged into $settingsFile"
        } catch {
            Write-Info "Could not auto-merge hooks into existing settings.json"
            Write-Info "Please manually add hooks from hooks.example.json to $settingsFile"
        }
    } else {
        $hooks | ConvertTo-Json -Depth 10 | Set-Content $settingsFile -Encoding UTF8
        Write-Ok "Created $settingsFile with hooks configuration"
    }
}

# Sync plugin cache — copy installed binary into Claude Code's plugin cache
function Sync-PluginCache {
    $cacheBase = Join-Path $env:USERPROFILE ".claude\plugins\cache\dullfig-plugins\memory-rlm"
    if (-not (Test-Path $cacheBase)) { return }

    $src = Join-Path $InstallDir $Binary
    if (-not (Test-Path $src)) { return }

    Get-ChildItem -Path $cacheBase -Directory | ForEach-Object {
        $binDir = Join-Path $_.FullName "bin"
        if (Test-Path $binDir) {
            $dest = Join-Path $binDir $Binary
            try {
                Copy-Item -Path $src -Destination $dest -Force
                Write-Ok "Synced plugin cache: $dest"
            } catch {
                Write-Info "Could not sync plugin cache at $dest (may be in use)"
            }
        }
    }
}

# Main
Write-Host ""
Write-Host "  ClaudeRLM Installer" -ForegroundColor Cyan
Write-Host "  Persistent project memory for Claude Code"
Write-Host ""

$version = Get-LatestVersion

if ($version) {
    try {
        Install-Binary -Version $version
    } catch {
        Write-Info "Binary download failed, trying cargo install..."
        Install-FromSource
    }
} else {
    Write-Info "No releases found, building from source..."
    Install-FromSource
}

Write-Ok "Installed memory-rlm to $InstallDir"
Write-Host ""

Add-ToPath
Configure-Hooks
Sync-PluginCache

Write-Host ""
Write-Ok "Done! Start a Claude Code session to see it in action."
Write-Host ""
Write-Host "  Optional: set CONTEXTMEM_LLM_API_KEY for LLM-enhanced distillation"
Write-Host "  See https://github.com/$Repo for details"
Write-Host ""
