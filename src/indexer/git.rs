use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use rusqlite::params;

use crate::db::Db;
use crate::indexer::{code, conversation};

/// Stats returned from a git catch-up operation.
#[derive(Debug, Default)]
pub struct CatchupStats {
    pub commits: usize,
    pub files_changed: usize,
}

/// Detect git changes since the last tracked commit and index them.
///
/// On first run, stores HEAD and returns early (no history dump).
/// On subsequent runs, indexes the commit log, changed files, and diff stats
/// as a `git_catchup` turn, then reindexes changed files for code structure.
///
/// All errors are non-fatal — this should never break non-git projects
/// or unusual git states (detached HEAD, shallow clones, etc.).
pub fn catchup(db: &Db, project_dir: &Path, session_id: &str) -> Result<CatchupStats> {
    // 1. Check if this is a git repo
    if !is_git_repo(project_dir) {
        return Ok(CatchupStats::default());
    }

    // 2. Get current HEAD
    let head = git_rev_parse_head(project_dir)
        .context("failed to get HEAD commit")?;

    if head.is_empty() {
        return Ok(CatchupStats::default());
    }

    // 3. Query last known commit
    let last_known = get_last_commit(db, project_dir)?;

    // 4. First run: store HEAD and return
    if last_known.is_none() {
        store_commit(db, project_dir, &head)?;
        return Ok(CatchupStats::default());
    }

    let last = last_known.unwrap();

    // 5. No changes
    if head == last {
        return Ok(CatchupStats::default());
    }

    // 6. Gather changes between last..HEAD
    let log = git_log_oneline(project_dir, &last, &head);
    let shortstat = git_diff_shortstat(project_dir, &last, &head);
    let changed_files = git_diff_name_only(project_dir, &last, &head);

    let commits: Vec<&str> = log.lines().collect();
    let commit_count = commits.len();
    let file_count = changed_files.len();

    // Build content for the turn
    let mut content = format!(
        "{} commits, {} files changed\n{}\n",
        commit_count,
        file_count,
        shortstat.trim(),
    );

    // Commit log
    if !commits.is_empty() {
        content.push_str("\nCommits:\n");
        for line in &commits {
            content.push_str(&format!("- {}\n", line));
        }
    }

    // Changed files
    if !changed_files.is_empty() {
        content.push_str("\nChanged files:\n");
        for f in &changed_files {
            content.push_str(&format!("- {}\n", f));
        }
    }

    // 7. Store as a conversation turn
    let file_refs: Vec<(String, String)> = changed_files
        .iter()
        .map(|f| (f.clone(), "git_change".to_string()))
        .collect();

    conversation::index_turn(
        db,
        session_id,
        "system",
        "git_catchup",
        &content,
        None,
        &file_refs,
    )?;

    // 8. Reindex changed files for code structure
    for file_name in &changed_files {
        let file_path = project_dir.join(file_name);
        if file_path.exists() {
            if let Err(e) = code::reindex_file(db, &file_path) {
                eprintln!("[memory-rlm] Git catch-up reindex failed for {}: {}", file_name, e);
            }
        }
    }

    // 9. Update stored HEAD
    store_commit(db, project_dir, &head)?;

    Ok(CatchupStats {
        commits: commit_count,
        files_changed: file_count,
    })
}

// --- Git CLI helpers ---

pub fn is_git_repo(dir: &Path) -> bool {
    Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .current_dir(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn git_rev_parse_head(dir: &Path) -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(dir)
        .output()
        .context("failed to run git rev-parse HEAD")?;

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_log_oneline(dir: &Path, from: &str, to: &str) -> String {
    let range = format!("{}..{}", from, to);
    Command::new("git")
        .args(["log", "--oneline", &range])
        .current_dir(dir)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
}

fn git_diff_shortstat(dir: &Path, from: &str, to: &str) -> String {
    let range = format!("{}..{}", from, to);
    Command::new("git")
        .args(["diff", "--shortstat", &range])
        .current_dir(dir)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
}

fn git_diff_name_only(dir: &Path, from: &str, to: &str) -> Vec<String> {
    let range = format!("{}..{}", from, to);
    Command::new("git")
        .args(["diff", "--name-only", &range])
        .current_dir(dir)
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect()
        })
        .unwrap_or_default()
}

// --- Database helpers ---

fn get_last_commit(db: &Db, project_dir: &Path) -> Result<Option<String>> {
    let conn = db.conn();
    let dir_str = project_dir.to_string_lossy();
    let result = conn.query_row(
        "SELECT last_commit_hash FROM git_state WHERE project_dir = ?1",
        params![dir_str.as_ref()],
        |row| row.get(0),
    );

    match result {
        Ok(hash) => Ok(Some(hash)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn store_commit(db: &Db, project_dir: &Path, hash: &str) -> Result<()> {
    let conn = db.conn();
    let dir_str = project_dir.to_string_lossy();
    conn.execute(
        "INSERT INTO git_state (project_dir, last_commit_hash, updated_at)
         VALUES (?1, ?2, datetime('now'))
         ON CONFLICT(project_dir)
         DO UPDATE SET last_commit_hash = ?2, updated_at = datetime('now')",
        params![dir_str.as_ref(), hash],
    )?;
    Ok(())
}
