use anyhow::Result;
use serde_json::json;

use crate::db::Db;
use crate::hooks::{self, HookInput};
use crate::indexer::{code, conversation, files, git, plans};
use crate::inject;

/// Handle SessionStart hook.
/// - source="startup": inject project memory (recent sessions, knowledge)
/// - source="compact": inject context relevant to current task
pub fn handle_start(input: &HookInput) -> Result<()> {
    let project_dir = hooks::project_dir(input);
    let session_id = hooks::session_id(input);
    let source = input.source.as_deref().unwrap_or("startup");

    let db = Db::open(std::path::Path::new(&project_dir))?;
    conversation::ensure_session(&db, &session_id, &project_dir)?;

    hooks::log_hook(&db, input, "SessionStart", &format!("source: {}", source));

    // Catch up on git changes since last session
    if source == "startup" {
        match git::catchup(&db, std::path::Path::new(&project_dir), &session_id) {
            Ok(stats) if stats.commits > 0 => {
                eprintln!(
                    "[memory-rlm] Git catch-up: {} commits, {} files changed",
                    stats.commits, stats.files_changed
                );
            }
            Err(e) => eprintln!("[memory-rlm] Git catch-up skipped: {}", e),
            _ => {}
        }
    }

    // File-hash catch-up for non-git projects
    if source == "startup" && !git::is_git_repo(std::path::Path::new(&project_dir)) {
        match files::catchup(&db, std::path::Path::new(&project_dir), &session_id) {
            Ok(stats) if stats.files_changed + stats.files_added + stats.files_deleted > 0 => {
                eprintln!(
                    "[memory-rlm] File catch-up: {} changed, {} added, {} deleted",
                    stats.files_changed, stats.files_added, stats.files_deleted
                );
            }
            Err(e) => eprintln!("[memory-rlm] File catch-up failed: {}", e),
            _ => {}
        }
    }

    // On startup, run initial code indexing if no index exists
    if source == "startup" && !code::has_index(&db)? {
        let dir = std::path::Path::new(&project_dir);
        if let Err(e) = code::index_project(&db, dir) {
            eprintln!("[memory-rlm] Initial code indexing failed: {}", e);
        }
    }

    let context = match source {
        "compact" => inject::build_compact_context(&db, &session_id)?,
        _ => inject::build_startup_context(&db)?,
    };

    // Print startup banner with quick stats
    if source == "startup" {
        let conn = db.conn();
        let sessions: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL",
            [], |row| row.get(0)
        ).unwrap_or(0);
        let symbols: i64 = conn.query_row(
            "SELECT COUNT(*) FROM symbols", [], |row| row.get(0)
        ).unwrap_or(0);
        let knowledge: i64 = conn.query_row(
            "SELECT COUNT(*) FROM knowledge WHERE superseded_by IS NULL",
            [], |row| row.get(0)
        ).unwrap_or(0);

        let active_plans: i64 = conn.query_row(
            "SELECT COUNT(*) FROM plans WHERE status IN ('created', 'in_progress')",
            [], |row| row.get(0)
        ).unwrap_or(0);

        let mut parts = Vec::new();
        if sessions > 0 { parts.push(format!("{} sessions", sessions)); }
        if symbols > 0 { parts.push(format!("{} symbols", symbols)); }
        if knowledge > 0 { parts.push(format!("{} knowledge", knowledge)); }
        if active_plans > 0 { parts.push(format!("{} active plan(s)", active_plans)); }

        if parts.is_empty() {
            eprintln!("[ClaudeRLM] Project memory initialized");
        } else {
            eprintln!("[ClaudeRLM] Project memory loaded ({})", parts.join(", "));
        }
    }

    if !context.is_empty() {
        let output = json!({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context
            }
        });
        println!("{}", serde_json::to_string(&output)?);
    }

    Ok(())
}

/// Handle SessionEnd hook: signal the MCP server to exit, then queue
/// deferred work. Returns instantly so Claude Code can proceed with shutdown.
/// The MCP server picks up the shutdown signal within ~300ms and exits
/// cleanly before Claude Code force-kills it.
pub fn handle_end(input: &HookInput) -> Result<()> {
    let project_dir = hooks::project_dir(input);
    let session_id = hooks::session_id(input);

    let db = Db::open(std::path::Path::new(&project_dir))?;

    hooks::log_hook(&db, input, "SessionEnd", "");

    // Signal the MCP server to exit gracefully via the task queue.
    // The server polls every 300ms, so it should exit before Claude Code
    // resorts to TerminateProcess.
    crate::db::tasks::enqueue_task(&db, "shutdown", &project_dir, None)?;

    // Evaluate plan completion before ending session
    if let Err(e) = plans::evaluate_completion(&db, &session_id) {
        eprintln!("[memory-rlm] Plan evaluation failed: {}", e);
    }
    if let Err(e) = plans::abandon_stale_plans(&db, 7) {
        eprintln!("[memory-rlm] Stale plan cleanup failed: {}", e);
    }

    // Queue distillation for the next session's MCP server
    crate::db::tasks::enqueue_task(&db, "distill_session", &project_dir, Some(&session_id))?;

    // Mark session as ended
    conversation::end_session(&db, &session_id, None)?;

    Ok(())
}

