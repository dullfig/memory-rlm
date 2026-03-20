use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;
use rusqlite::params;

use crate::db::Db;
use crate::indexer::{code, conversation};

/// Stats returned from a file-hash catch-up operation.
#[derive(Debug, Default)]
pub struct CatchupStats {
    pub files_changed: usize,
    pub files_added: usize,
    pub files_deleted: usize,
}

/// Detect file changes since the last session using content hashes.
///
/// On first run, stores all hashes and returns early (no "everything changed" dump).
/// On subsequent runs, compares hashes to find changed, added, and deleted files,
/// logs them as a `file_catchup` turn, and reindexes changed code files.
pub fn catchup(db: &Db, project_dir: &Path, session_id: &str) -> Result<CatchupStats> {
    let dir_str = project_dir.to_string_lossy().to_string();

    // 1. Walk project files and compute current hashes
    let current_hashes = walk_and_hash(project_dir)?;

    // 2. Query stored hashes for this project
    let stored_hashes = get_stored_hashes(db, &dir_str)?;

    // 3. First run: store all hashes and return early
    if stored_hashes.is_empty() {
        store_all_hashes(db, &dir_str, &current_hashes)?;
        return Ok(CatchupStats::default());
    }

    // 4. Compare to find changes
    let mut changed: Vec<String> = Vec::new();
    let mut added: Vec<String> = Vec::new();
    let mut deleted: Vec<String> = Vec::new();

    for (path, hash) in &current_hashes {
        match stored_hashes.get(path) {
            Some(old_hash) if old_hash != hash => changed.push(path.clone()),
            None => added.push(path.clone()),
            _ => {}
        }
    }

    for path in stored_hashes.keys() {
        if !current_hashes.contains_key(path) {
            deleted.push(path.clone());
        }
    }

    let stats = CatchupStats {
        files_changed: changed.len(),
        files_added: added.len(),
        files_deleted: deleted.len(),
    };

    if stats.files_changed + stats.files_added + stats.files_deleted == 0 {
        return Ok(stats);
    }

    // 5. Build content summary
    let mut content = format!(
        "{} files changed, {} added, {} deleted\n",
        stats.files_changed, stats.files_added, stats.files_deleted
    );

    if !changed.is_empty() {
        content.push_str("\nChanged files:\n");
        for f in &changed {
            content.push_str(&format!("- {}\n", f));
        }
    }

    if !added.is_empty() {
        content.push_str("\nAdded files:\n");
        for f in &added {
            content.push_str(&format!("- {}\n", f));
        }
    }

    if !deleted.is_empty() {
        content.push_str("\nDeleted files:\n");
        for f in &deleted {
            content.push_str(&format!("- {}\n", f));
        }
    }

    // 6. Store as a conversation turn
    let mut file_refs: Vec<(String, String)> = Vec::new();
    for f in &changed {
        file_refs.push((f.clone(), "file_change".to_string()));
    }
    for f in &added {
        file_refs.push((f.clone(), "file_add".to_string()));
    }
    for f in &deleted {
        file_refs.push((f.clone(), "file_delete".to_string()));
    }

    conversation::index_turn(
        db,
        session_id,
        "system",
        "file_catchup",
        &content,
        None,
        &file_refs,
    )?;

    // 7. Reindex changed/added code files
    for file_name in changed.iter().chain(added.iter()) {
        let file_path = project_dir.join(file_name);
        if file_path.exists() {
            if let Err(e) = code::reindex_file(db, &file_path) {
                eprintln!(
                    "[memory-rlm] File catch-up reindex failed for {}: {}",
                    file_name, e
                );
            }
        }
    }

    // 8. Remove deleted files from file_hashes and symbols
    //    Scoped so conn is dropped before store_all_hashes acquires it
    {
        let conn = db.conn();
        for file_name in &deleted {
            let full_path = project_dir.join(file_name).to_string_lossy().to_string();
            conn.execute(
                "DELETE FROM file_hashes WHERE project_dir = ?1 AND file_path = ?2",
                params![&dir_str, file_name],
            )?;
            conn.execute(
                "DELETE FROM symbols WHERE file_path = ?1",
                params![&full_path],
            )?;
        }
    }

    // 9. Update file_hashes with new hashes
    store_all_hashes(db, &dir_str, &current_hashes)?;

    Ok(stats)
}

/// Compute a SipHash of file contents. Returns hex-encoded u64.
fn hash_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

/// Walk the project directory and compute hashes for all files.
/// Uses the same walker pattern as code::index_project — respects .gitignore,
/// skips hidden dirs and node_modules/target/etc.
fn walk_and_hash(project_dir: &Path) -> Result<HashMap<String, String>> {
    let mut hashes = HashMap::new();

    let walker = WalkBuilder::new(project_dir)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                "node_modules" | "target" | "dist" | "build" | ".git" | "__pycache__" | "vendor"
                    | ".venv" | "venv"
            )
        })
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }

        let path = entry.path();

        // Compute relative path for storage
        let rel_path = match path.strip_prefix(project_dir) {
            Ok(rel) => rel.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        match hash_file(path) {
            Ok(hash) => {
                hashes.insert(rel_path, hash);
            }
            Err(_) => continue, // Skip files we can't read
        }
    }

    Ok(hashes)
}

/// Get all stored file hashes for a project.
fn get_stored_hashes(db: &Db, project_dir: &str) -> Result<HashMap<String, String>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT file_path, content_hash FROM file_hashes WHERE project_dir = ?1",
    )?;

    let rows: Vec<(String, String)> = stmt
        .query_map(params![project_dir], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows.into_iter().collect())
}

/// Store all file hashes for a project (upsert).
fn store_all_hashes(db: &Db, project_dir: &str, hashes: &HashMap<String, String>) -> Result<()> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "INSERT INTO file_hashes (project_dir, file_path, content_hash, updated_at)
         VALUES (?1, ?2, ?3, datetime('now'))
         ON CONFLICT(project_dir, file_path)
         DO UPDATE SET content_hash = ?3, updated_at = datetime('now')",
    )?;

    for (path, hash) in hashes {
        stmt.execute(params![project_dir, path, hash])?;
    }

    Ok(())
}
