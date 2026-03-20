use anyhow::Result;

use crate::db::Db;
use crate::hooks::{self, HookInput};

/// Handle PreCompact hook: ensure all context is indexed before compaction.
///
/// This is critical — compaction will compress/discard older context, so we
/// need to make sure everything valuable has been captured in the index.
///
/// Re-indexing stale code files is delegated to a background task (picked up by
/// the MCP server's task poller) so this hook returns quickly.
pub fn handle(input: &HookInput) -> Result<()> {
    let project_dir = hooks::project_dir(input);
    let session_id = hooks::session_id(input);
    let db = Db::open(std::path::Path::new(&project_dir))?;

    hooks::log_hook(&db, input, "PreCompact", "");

    eprintln!("[memory-rlm] PreCompact: ensuring index is current for session {session_id}");

    // 1. Generate checkpoint summary — this is the critical part that
    //    survives compaction. Must complete before compaction proceeds.
    generate_checkpoint_summary(&db, &session_id)?;
    eprintln!("[memory-rlm] PreCompact: checkpoint saved");

    // 2. Enqueue stale-file re-indexing as a background task for the MCP server.
    //    This avoids blocking compaction on potentially slow tree-sitter work.
    crate::db::tasks::enqueue_task(&db, "reindex_stale", &project_dir, None)?;
    eprintln!("[memory-rlm] PreCompact: enqueued reindex_stale task");

    Ok(())
}

/// Generate a checkpoint summary of the session so far and store it as a special turn.
fn generate_checkpoint_summary(db: &Db, session_id: &str) -> Result<()> {
    let conn = db.conn();

    // Gather stats
    let turn_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM turns WHERE session_id = ?1",
        [session_id],
        |row| row.get(0),
    )?;

    if turn_count == 0 {
        return Ok(());
    }

    // Get all user requests
    let mut stmt = conn.prepare(
        "SELECT content FROM turns WHERE session_id = ?1 AND turn_type = 'request'
         ORDER BY turn_number ASC",
    )?;
    let requests: Vec<String> = stmt
        .query_map([session_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    // Get files modified
    let mut stmt = conn.prepare(
        "SELECT DISTINCT tf.file_path, tf.action
         FROM turn_files tf
         JOIN turns t ON t.id = tf.turn_id
         WHERE t.session_id = ?1 AND tf.action IN ('edit', 'write', 'create')
         ORDER BY tf.file_path",
    )?;
    let modified_files: Vec<(String, String)> = stmt
        .query_map([session_id], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(|r| r.ok())
        .collect();

    // Get edit summaries (most recent edits, condensed)
    let mut stmt = conn.prepare(
        "SELECT content FROM turns WHERE session_id = ?1 AND turn_type = 'code_edit'
         ORDER BY turn_number DESC LIMIT 10",
    )?;
    let recent_edits: Vec<String> = stmt
        .query_map([session_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    // Build the checkpoint summary
    let mut summary = String::from("[Pre-Compaction Checkpoint]\n");

    summary.push_str("Tasks:\n");
    for (i, req) in requests.iter().enumerate() {
        let truncated = if req.len() > 200 {
            let end = req.floor_char_boundary(200);
            format!("{}...", &req[..end])
        } else {
            req.clone()
        };
        summary.push_str(&format!("  {}. {}\n", i + 1, truncated));
    }

    if !modified_files.is_empty() {
        summary.push_str("\nFiles modified:\n");
        for (path, action) in &modified_files {
            summary.push_str(&format!("  - {} ({})\n", path, action));
        }
    }

    if !recent_edits.is_empty() {
        summary.push_str("\nRecent edits:\n");
        for edit in recent_edits.iter().rev() {
            let truncated = if edit.len() > 300 {
                let end = edit.floor_char_boundary(300);
                format!("{}...", &edit[..end])
            } else {
                edit.clone()
            };
            summary.push_str(&format!("  - {}\n", truncated));
        }
    }

    // Store as a special "checkpoint" turn
    crate::indexer::conversation::index_turn(
        db,
        session_id,
        "system",
        "checkpoint",
        &summary,
        None,
        &[],
    )?;

    Ok(())
}
