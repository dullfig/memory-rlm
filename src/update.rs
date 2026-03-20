//! Auto-update mechanism for memory-rlm.
//!
//! Two-phase update: **check** → **apply** (immediate).
//!
//! - On MCP startup, apply any legacy `.staged` update (migration from old versions).
//! - In the background, check GitHub for a newer release and apply it immediately.
//! - The binary can be renamed while running on Windows, so no restart-to-apply needed.
//! - Archives also contain hooks/hooks.json which gets installed alongside the binary.

use std::path::{Path, PathBuf};

const REPO: &str = "dullfig/memory-rlm";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Contents extracted from a release archive.
struct ArchiveContents {
    binary: Vec<u8>,
    hooks_json: Option<Vec<u8>>,
}

/// Apply a previously staged update (migration from old version).
///
/// If `<binary>.staged` exists, promotes it via rename:
/// - Windows: rename self → `.old`, rename `.staged` → self
/// - Unix: atomic rename `.staged` → self
///
/// Returns true if an update was applied.
pub fn apply_staged_update() -> bool {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return false,
    };

    let staged = suffixed_path(&exe, ".staged");
    if !staged.exists() {
        return false;
    }

    tracing::info!("Found staged update at {}, applying...", staged.display());

    #[cfg(windows)]
    {
        let old = suffixed_path(&exe, ".old");
        if let Err(e) = std::fs::rename(&exe, &old) {
            tracing::warn!("Failed to rename current binary to .old: {}", e);
            return false;
        }
        if let Err(e) = std::fs::rename(&staged, &exe) {
            tracing::warn!("Failed to rename staged binary: {}", e);
            // Try to restore
            let _ = std::fs::rename(&old, &exe);
            return false;
        }
        tracing::info!("Update applied successfully (v{})", CURRENT_VERSION);
        true
    }

    #[cfg(not(windows))]
    {
        if let Err(e) = std::fs::rename(&staged, &exe) {
            tracing::warn!("Failed to apply staged update: {}", e);
            return false;
        }
        tracing::info!("Update applied successfully");
        true
    }
}

/// Remove leftover files from previous updates (best-effort).
///
/// Cleans up: `<exe>.old`, `<exe>.staged`, `<exe>.tmp`,
/// and `hooks.json.old`/`hooks.json.tmp` under the plugin hooks dir.
pub fn cleanup_old_files() {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };

    for suffix in &[".old", ".staged", ".tmp"] {
        let path = suffixed_path(&exe, suffix);
        if path.exists() {
            match std::fs::remove_file(&path) {
                Ok(()) => tracing::debug!("Removed {}", path.display()),
                Err(e) => tracing::debug!("Could not remove {}: {} (may be locked)", path.display(), e),
            }
        }
    }

    // Clean up hooks.json leftovers
    if let Some(root) = plugin_root_from_exe(&exe) {
        let hooks_dir = root.join("hooks");
        for name in &["hooks.json.old", "hooks.json.tmp"] {
            let path = hooks_dir.join(name);
            if path.exists() {
                match std::fs::remove_file(&path) {
                    Ok(()) => tracing::debug!("Removed {}", path.display()),
                    Err(e) => tracing::debug!("Could not remove {}: {}", path.display(), e),
                }
            }
        }
    }
}

/// Spawn a background update check. Non-blocking — returns immediately.
pub fn spawn_update_check() {
    if is_auto_update_disabled() {
        tracing::debug!("Auto-update disabled, skipping check");
        return;
    }

    tokio::spawn(async {
        if let Err(e) = check_and_apply_update().await {
            tracing::debug!("Update check failed: {}", e);
        }
    });
}

/// Check GitHub for a newer release and apply it immediately if found.
async fn check_and_apply_update() -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;

    if should_skip_check(&exe) {
        tracing::debug!("Update check throttled, skipping");
        return Ok(());
    }

    tracing::info!("Checking for updates...");

    let client = reqwest::Client::builder()
        .user_agent(format!("memory-rlm/{}", CURRENT_VERSION))
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let url = format!("https://api.github.com/repos/{}/releases/latest", REPO);
    let resp = client.get(&url).send().await?;

    if !resp.status().is_success() {
        record_check(&exe);
        anyhow::bail!("GitHub API returned {}", resp.status());
    }

    let release: GitHubRelease = resp.json().await?;
    let remote_version = release.tag_name.trim_start_matches('v');

    if !is_newer(remote_version, CURRENT_VERSION) {
        tracing::info!("Up to date (v{})", CURRENT_VERSION);
        record_check(&exe);
        return Ok(());
    }

    tracing::info!(
        "New version available: v{} (current: v{})",
        remote_version,
        CURRENT_VERSION
    );

    let target = platform_target();
    if target == "unknown" {
        record_check(&exe);
        anyhow::bail!("No pre-built binary for this platform");
    }

    let asset_name = if cfg!(windows) {
        format!("memory-rlm-{}.zip", target)
    } else {
        format!("memory-rlm-{}.tar.gz", target)
    };

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| anyhow::anyhow!("No asset found for {}", asset_name))?;

    tracing::info!("Downloading {}...", asset.name);
    let resp = client.get(&asset.browser_download_url).send().await?;
    if !resp.status().is_success() {
        record_check(&exe);
        anyhow::bail!("Asset download returned {}", resp.status());
    }

    let bytes = resp.bytes().await?;
    let contents = extract_archive_contents(&bytes)?;
    validate_binary(&contents.binary)?;

    // Apply binary immediately
    apply_binary_now(&exe, &contents.binary)?;

    tracing::info!(
        "Applied update v{} → v{} to {}",
        CURRENT_VERSION,
        remote_version,
        exe.display()
    );

    // Write version marker for next-session notification
    let marker = suffixed_path(&exe, ".updated");
    let _ = std::fs::write(&marker, remote_version);

    // Apply hooks.json if present in the archive
    if let Some(hooks_data) = &contents.hooks_json {
        if let Some(root) = plugin_root_from_exe(&exe) {
            let hooks_path = root.join("hooks").join("hooks.json");
            let root_str = root.to_string_lossy();
            match apply_hooks_now(&hooks_path, hooks_data, &root_str) {
                Ok(()) => tracing::info!("Updated hooks.json at {}", hooks_path.display()),
                Err(e) => tracing::warn!("Failed to update hooks.json: {}", e),
            }
        }
    }

    // Clean up any leftover .staged file from old versions
    let staged = suffixed_path(&exe, ".staged");
    if staged.exists() {
        let _ = std::fs::remove_file(&staged);
    }

    record_check(&exe);
    Ok(())
}

/// Check if the binary was updated in a previous session.
/// Returns the new version string if an `.updated` marker exists, then deletes it.
pub fn check_version_updated() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let marker = suffixed_path(&exe, ".updated");
    let version = std::fs::read_to_string(&marker).ok()?;
    let _ = std::fs::remove_file(&marker);
    let version = version.trim().to_string();
    if version.is_empty() { None } else { Some(version) }
}

// ---------------------------------------------------------------------------
// Apply helpers
// ---------------------------------------------------------------------------

/// Apply binary data to the exe path immediately via rename dance.
fn apply_binary_now(exe: &Path, data: &[u8]) -> anyhow::Result<()> {
    let tmp = suffixed_path(exe, ".tmp");
    std::fs::write(&tmp, data)?;

    #[cfg(not(windows))]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }

    #[cfg(windows)]
    {
        let old = suffixed_path(exe, ".old");
        // Remove stale .old if it exists (from a previous update)
        let _ = std::fs::remove_file(&old);
        if let Err(e) = std::fs::rename(exe, &old) {
            let _ = std::fs::remove_file(&tmp);
            anyhow::bail!("Failed to rename current binary to .old: {}", e);
        }
        if let Err(e) = std::fs::rename(&tmp, exe) {
            // Try to restore
            let _ = std::fs::rename(&old, exe);
            anyhow::bail!("Failed to rename .tmp to binary: {}", e);
        }
    }

    #[cfg(not(windows))]
    {
        std::fs::rename(&tmp, exe)?;
    }

    Ok(())
}

/// Apply hooks.json data with `${CLAUDE_PLUGIN_ROOT}` resolution.
fn apply_hooks_now(hooks_path: &Path, data: &[u8], plugin_root: &str) -> anyhow::Result<()> {
    let raw = std::str::from_utf8(data)?;
    let resolved = raw.replace("${CLAUDE_PLUGIN_ROOT}", plugin_root);

    // Validate it's still valid JSON after resolution
    let _: serde_json::Value = serde_json::from_str(&resolved)
        .map_err(|e| anyhow::anyhow!("hooks.json invalid after resolution: {}", e))?;

    // Ensure hooks directory exists
    if let Some(parent) = hooks_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let tmp = hooks_path.with_extension("json.tmp");
    std::fs::write(&tmp, &resolved)?;

    #[cfg(windows)]
    {
        let old = hooks_path.with_extension("json.old");
        let _ = std::fs::remove_file(&old);
        if hooks_path.exists() {
            if let Err(e) = std::fs::rename(hooks_path, &old) {
                let _ = std::fs::remove_file(&tmp);
                anyhow::bail!("Failed to rename current hooks.json to .old: {}", e);
            }
        }
        if let Err(e) = std::fs::rename(&tmp, hooks_path) {
            // Try to restore
            if old.exists() {
                let _ = std::fs::rename(&old, hooks_path);
            }
            anyhow::bail!("Failed to rename .tmp to hooks.json: {}", e);
        }
    }

    #[cfg(not(windows))]
    {
        std::fs::rename(&tmp, hooks_path)?;
    }

    Ok(())
}

/// Derive the plugin root directory from the exe path.
///
/// Plugin layout: `{root}/bin/memory-rlm[.exe]`
/// So plugin root = exe's parent's parent.
pub fn plugin_root_from_exe(exe: &Path) -> Option<PathBuf> {
    let bin_dir = exe.parent()?;
    let bin_name = bin_dir.file_name()?.to_str()?;
    if bin_name == "bin" {
        Some(bin_dir.parent()?.to_path_buf())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// General helpers
// ---------------------------------------------------------------------------

/// Append a suffix to the full path (e.g. `/foo/bar.exe` + `.staged` → `/foo/bar.exe.staged`).
fn suffixed_path(exe: &Path, suffix: &str) -> PathBuf {
    let mut p = exe.as_os_str().to_owned();
    p.push(suffix);
    PathBuf::from(p)
}

/// Returns true if we checked within the last hour (touch-file throttle).
fn should_skip_check(exe: &Path) -> bool {
    let stamp = suffixed_path(exe, ".update-check");
    stamp
        .metadata()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.elapsed().ok())
        .map(|d| d.as_secs() < 3600)
        .unwrap_or(false)
}

/// Record that we performed an update check (touch the stamp file).
fn record_check(exe: &Path) {
    let stamp = suffixed_path(exe, ".update-check");
    let _ = std::fs::write(&stamp, "");
}

/// Check if auto-update is disabled via env var or config.
fn is_auto_update_disabled() -> bool {
    if std::env::var("CLAUDE_RLM_NO_UPDATE").ok().as_deref() == Some("1") {
        return true;
    }

    // Check [update] section in config TOML
    if let Some(path) = crate::llm::global_config_path() {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Ok(doc) = contents.parse::<toml::Table>() {
                if let Some(update) = doc.get("update").and_then(|v| v.as_table()) {
                    if let Some(enabled) = update.get("auto_update").and_then(|v| v.as_bool()) {
                        return !enabled;
                    }
                }
            }
        }
    }

    false
}

/// Parse a semver string into (major, minor, patch).
fn parse_semver(s: &str) -> Option<(u32, u32, u32)> {
    let s = s.trim_start_matches('v');
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    Some((
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    ))
}

/// Returns true if `remote` is strictly newer than `local`.
fn is_newer(remote: &str, local: &str) -> bool {
    match (parse_semver(remote), parse_semver(local)) {
        (Some(r), Some(l)) => r > l,
        _ => false,
    }
}

/// Return the Rust target triple for the current platform.
fn platform_target() -> &'static str {
    if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
        "x86_64-pc-windows-msvc"
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        "x86_64-apple-darwin"
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        "aarch64-apple-darwin"
    } else {
        "unknown"
    }
}

/// Validate that the binary data looks like a real executable.
fn validate_binary(data: &[u8]) -> anyhow::Result<()> {
    if data.len() < 4 {
        anyhow::bail!("Binary too small ({} bytes)", data.len());
    }

    if cfg!(windows) {
        if &data[..2] != b"MZ" {
            anyhow::bail!("Invalid Windows binary (missing MZ header)");
        }
    } else {
        let is_elf = &data[..4] == b"\x7fELF";
        let is_macho = data[..4] == [0xfe, 0xed, 0xfa, 0xce]
            || data[..4] == [0xfe, 0xed, 0xfa, 0xcf]
            || data[..4] == [0xce, 0xfa, 0xed, 0xfe]
            || data[..4] == [0xcf, 0xfa, 0xed, 0xfe];
        if !is_elf && !is_macho {
            anyhow::bail!("Invalid binary (not ELF or Mach-O)");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Archive extraction
// ---------------------------------------------------------------------------

/// Extract binary and optional hooks.json from a zip archive (Windows).
#[cfg(windows)]
fn extract_archive_contents(data: &[u8]) -> anyhow::Result<ArchiveContents> {
    use std::io::Read;

    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let mut binary: Option<Vec<u8>> = None;
    let mut hooks_json: Option<Vec<u8>> = None;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();

        if name == "memory-rlm.exe" || name.ends_with("/memory-rlm.exe") {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            binary = Some(buf);
        } else if name == "hooks/hooks.json" || name.ends_with("/hooks/hooks.json") {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            hooks_json = Some(buf);
        }
    }

    Ok(ArchiveContents {
        binary: binary.ok_or_else(|| anyhow::anyhow!("memory-rlm.exe not found in zip archive"))?,
        hooks_json,
    })
}

/// Extract binary and optional hooks.json from a tar.gz archive (Unix).
#[cfg(not(windows))]
fn extract_archive_contents(data: &[u8]) -> anyhow::Result<ArchiveContents> {
    use std::io::Read;

    let gz = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(gz);

    let mut binary: Option<Vec<u8>> = None;
    let mut hooks_json: Option<Vec<u8>> = None;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let path_str = path.to_string_lossy();

        if path_str == "memory-rlm" || path_str.ends_with("/memory-rlm") {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            binary = Some(buf);
        } else if path_str == "hooks/hooks.json" || path_str.ends_with("/hooks/hooks.json") {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            hooks_json = Some(buf);
        }
    }

    Ok(ArchiveContents {
        binary: binary.ok_or_else(|| anyhow::anyhow!("memory-rlm not found in tar.gz archive"))?,
        hooks_json,
    })
}

// ---------------------------------------------------------------------------
// GitHub API types
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(serde::Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_semver() {
        assert_eq!(parse_semver("0.2.0"), Some((0, 2, 0)));
        assert_eq!(parse_semver("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_semver("1.0"), None);
        assert_eq!(parse_semver("abc"), None);
    }

    #[test]
    fn test_is_newer() {
        assert!(is_newer("0.3.0", "0.2.0"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(!is_newer("0.2.0", "0.2.0"));
        assert!(!is_newer("0.1.0", "0.2.0"));
        assert!(!is_newer("v0.2.0", "0.2.0"));
        assert!(is_newer("v0.3.0", "0.2.0"));
    }

    #[test]
    fn test_platform_target() {
        let target = platform_target();
        assert_ne!(target, "unknown");
    }

    #[test]
    fn test_plugin_root_from_exe() {
        // Windows-style path
        let exe = Path::new("C:/Users/Dan/plugins/memory-rlm/0.2.0/bin/memory-rlm.exe");
        let root = plugin_root_from_exe(exe).unwrap();
        assert_eq!(root, Path::new("C:/Users/Dan/plugins/memory-rlm/0.2.0"));

        // Unix-style path
        let exe = Path::new("/home/user/.claude/plugins/cache/memory-rlm/0.2.0/bin/memory-rlm");
        let root = plugin_root_from_exe(exe).unwrap();
        assert_eq!(
            root,
            Path::new("/home/user/.claude/plugins/cache/memory-rlm/0.2.0")
        );

        // Not in a bin/ directory — should return None
        let exe = Path::new("/usr/local/memory-rlm");
        assert!(plugin_root_from_exe(exe).is_none());
    }

    #[test]
    fn test_hooks_json_resolution() {
        let raw = r#"{"hooks":{"SessionStart":[{"hooks":[{"command":"${CLAUDE_PLUGIN_ROOT}/bin/memory-rlm session-start"}]}]}}"#;
        let resolved = raw.replace("${CLAUDE_PLUGIN_ROOT}", "/home/user/.claude/plugins/cache/memory-rlm/0.2.0");

        // Should still be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&resolved).unwrap();
        let cmd = parsed["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert!(cmd.starts_with("/home/user/"));
        assert!(cmd.contains("memory-rlm session-start"));
        assert!(!cmd.contains("${CLAUDE_PLUGIN_ROOT}"));
    }
}
