pub mod schema;
pub mod search;
pub mod tasks;

use anyhow::Result;
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Thread-safe database handle.
#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl Db {
    /// Open (or create) the ClaudeRLM database at the given project directory.
    /// Database lives at `<project_dir>/.claude/memory-rlm.db`.
    pub fn open(project_dir: &Path) -> Result<Self> {
        let db_dir = project_dir.join(".claude");
        std::fs::create_dir_all(&db_dir)?;
        let db_path = db_dir.join("memory-rlm.db");
        let conn = Connection::open(&db_path)?;

        // Enable WAL mode for better concurrent access
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            path: db_path,
        };
        db.run_migrations()?;
        Ok(db)
    }

    /// Open an in-memory database (for testing).
    #[allow(dead_code)]
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            path: PathBuf::from(":memory:"),
        };
        db.run_migrations()?;
        Ok(db)
    }

    /// Get a lock on the underlying connection.
    pub fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("db mutex poisoned")
    }

    /// Get the project directory (parent of `.claude/`).
    pub fn project_dir(&self) -> String {
        self.path
            .parent() // .claude/
            .and_then(|p| p.parent()) // project_dir
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn run_migrations(&self) -> Result<()> {
        schema::create_tables(&self.conn())
    }
}
