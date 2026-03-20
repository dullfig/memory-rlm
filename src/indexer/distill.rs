use anyhow::Result;
use rusqlite::params;
use serde::Deserialize;

use crate::db::Db;
use crate::db::search;
use crate::llm::LlmConfig;

/// Distill knowledge from a completed session using LLM if available,
/// falling back to heuristic extraction.
pub fn distill_session_smart(db: &Db, session_id: &str) -> Result<DistillStats> {
    if let Some(llm) = LlmConfig::from_env() {
        eprintln!("[memory-rlm] Using LLM distillation ({:?}, model={})", llm.provider, llm.model);
        match distill_session_llm(db, session_id, &llm) {
            Ok(stats) => {
                eprintln!("[memory-rlm] LLM distillation extracted {} entries", stats.extracted);
                return Ok(stats);
            }
            Err(e) => {
                eprintln!("[memory-rlm] LLM distillation failed, falling back to heuristics: {}", e);
            }
        }
    } else {
        eprintln!("[memory-rlm] No LLM configured, using heuristic distillation");
    }
    distill_session(db, session_id)
}

/// Distill knowledge using an LLM for high-quality extraction.
fn distill_session_llm(db: &Db, session_id: &str, llm: &LlmConfig) -> Result<DistillStats> {
    let conn = db.conn();
    let turns = search::session_turns(&conn, session_id)?;
    drop(conn);

    if turns.is_empty() {
        return Ok(DistillStats::default());
    }

    // Build a condensed transcript for the LLM
    let mut transcript = String::new();
    for turn in &turns {
        let label = match turn.turn_type.as_str() {
            "request" => "USER",
            "code_edit" => "EDIT",
            "file_read" => "READ",
            "bash_cmd" => "BASH",
            "checkpoint" => "CHECKPOINT",
            _ => "OTHER",
        };
        let content = truncate(&turn.content, 500);
        let files_str = if !turn.files.is_empty() {
            format!(" [files: {}]", turn.files.join(", "))
        } else {
            String::new()
        };
        transcript.push_str(&format!("[{}]{} {}\n", label, files_str, content));
    }

    // Cap transcript to ~12K chars (~3K tokens) to keep costs minimal
    let transcript = truncate(&transcript, 12_000);

    let system_prompt = r#"You are a knowledge extraction system. Analyze the conversation transcript and extract important knowledge entries.

For each entry, determine:
- category: one of "decision", "preference", "convention", "pattern", "bug_fix", "architecture", "debugging_insight"
- subject: a short label (2-6 words) describing what the knowledge is about
- content: a concise description (1-3 sentences) of the knowledge
- confidence: 0.0-1.0 indicating how confident you are this is real, persistent knowledge (not just a one-off)

Focus on:
- Technology/library decisions and WHY they were chosen
- User preferences and coding conventions (explicit or implied)
- Bug fixes and their root causes (so they can be avoided in future)
- Architectural patterns established in this session
- Debugging insights that would help future sessions

Do NOT extract:
- Trivial facts (file was read, command was run)
- One-time task details that won't be relevant later
- Information that is obvious from the code itself

Respond with ONLY a JSON array. No markdown, no explanation. Example:
[
  {"category": "decision", "subject": "auth strategy", "content": "Chose JWT for API auth because the app is stateless and needs to support multiple frontends.", "confidence": 0.9},
  {"category": "bug_fix", "subject": "race condition in cache", "content": "Cache invalidation had a TOCTOU race. Fixed by using atomic compare-and-swap.", "confidence": 0.85}
]

If no meaningful knowledge can be extracted, respond with: []"#;

    let response = llm.complete(system_prompt, &transcript)?;

    // Parse the JSON response — be lenient about markdown wrappers
    let json_str = extract_json_array(&response);
    let entries: Vec<LlmKnowledgeEntry> = serde_json::from_str(json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse LLM response as JSON: {}. Response: {}", e, &response[..response.len().min(500)]))?;

    let mut stats = DistillStats::default();

    for entry in &entries {
        // Validate category
        let valid_categories = [
            "decision", "preference", "convention", "pattern",
            "bug_fix", "architecture", "debugging_insight",
        ];
        if !valid_categories.contains(&entry.category.as_str()) {
            continue;
        }

        // Clamp confidence to valid range
        let confidence = entry.confidence.clamp(0.1, 1.0);

        stats.extracted += upsert_knowledge(
            db,
            session_id,
            &entry.category,
            &entry.subject,
            &entry.content,
            confidence,
        )?;
    }

    Ok(stats)
}

/// Extract a JSON array from a response that might have markdown code fences.
fn extract_json_array(s: &str) -> &str {
    let s = s.trim();
    // Strip ```json ... ``` wrapper
    let s = s.strip_prefix("```json").or_else(|| s.strip_prefix("```")).unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim()
}

#[derive(Deserialize)]
struct LlmKnowledgeEntry {
    category: String,
    subject: String,
    content: String,
    confidence: f64,
}

/// Distill knowledge from a completed session using heuristic pattern matching.
pub fn distill_session(db: &Db, session_id: &str) -> Result<DistillStats> {
    let conn = db.conn();
    let turns = search::session_turns(&conn, session_id)?;
    drop(conn);

    if turns.is_empty() {
        return Ok(DistillStats::default());
    }

    let mut stats = DistillStats::default();

    // 1. Extract decisions from user requests and edits
    let requests: Vec<&search::TurnSearchResult> = turns
        .iter()
        .filter(|t| t.turn_type == "request")
        .collect();

    let edits: Vec<&search::TurnSearchResult> = turns
        .iter()
        .filter(|t| t.turn_type == "code_edit")
        .collect();

    // 2. Detect technology/library decisions
    //    Pattern: user mentions a specific technology or the edit introduces an import
    for req in &requests {
        let content_lower = req.content.to_lowercase();

        // Technology decision patterns
        let tech_patterns = [
            ("jwt", "authentication", "decision"),
            ("bcrypt", "password hashing", "decision"),
            ("oauth", "authentication", "decision"),
            ("redis", "caching", "decision"),
            ("postgres", "database", "decision"),
            ("sqlite", "database", "decision"),
            ("mongodb", "database", "decision"),
            ("graphql", "API", "decision"),
            ("rest", "API", "decision"),
            ("grpc", "API", "decision"),
            ("docker", "deployment", "decision"),
            ("kubernetes", "deployment", "decision"),
            ("webpack", "bundling", "decision"),
            ("vite", "bundling", "decision"),
            ("tokio", "async runtime", "decision"),
            ("actix", "web framework", "decision"),
            ("axum", "web framework", "decision"),
            ("express", "web framework", "decision"),
            ("fastapi", "web framework", "decision"),
            ("django", "web framework", "decision"),
            ("react", "UI framework", "decision"),
            ("vue", "UI framework", "decision"),
            ("svelte", "UI framework", "decision"),
        ];

        for (keyword, area, category) in &tech_patterns {
            if content_lower.contains(keyword) {
                let subject = format!("{} choice", area);
                let content = format!(
                    "User chose {} for {}. Context: {}",
                    keyword,
                    area,
                    truncate(&req.content, 200)
                );
                stats.extracted += upsert_knowledge(
                    db, session_id, category, &subject, &content, 0.7,
                )?;
            }
        }

        // Convention/preference patterns
        if content_lower.contains("always ") || content_lower.contains("never ") {
            let subject = extract_preference_subject(&content_lower);
            let content = truncate(&req.content, 300);
            stats.extracted += upsert_knowledge(
                db, session_id, "preference", &subject, &content, 0.8,
            )?;
        }

        // Pattern: "use X instead of Y" or "prefer X over Y"
        if content_lower.contains("instead of")
            || content_lower.contains("prefer")
            || content_lower.contains("rather than")
        {
            let subject = "coding preference";
            let content = truncate(&req.content, 300);
            stats.extracted += upsert_knowledge(
                db, session_id, "preference", subject, &content, 0.7,
            )?;
        }
    }

    // 3. Detect patterns from code edits
    //    Analyze what files were edited and what kinds of changes were made
    let mut files_changed: Vec<String> = Vec::new();
    for edit in &edits {
        files_changed.extend(edit.files.iter().cloned());

        // Detect bug fix patterns
        let content_lower = edit.content.to_lowercase();
        if content_lower.contains("fix") || content_lower.contains("bug") {
            let subject = if !edit.files.is_empty() {
                format!("bug fix in {}", short_path(&edit.files[0]))
            } else {
                "bug fix".to_string()
            };
            let content = truncate(&edit.content, 300);
            stats.extracted += upsert_knowledge(
                db, session_id, "bug_fix", &subject, &content, 0.8,
            )?;
        }
    }

    // 4. Detect architectural patterns from file structure
    files_changed.sort();
    files_changed.dedup();

    if files_changed.len() >= 3 {
        // Multiple files touched suggests a cross-cutting concern
        let subject = "session scope";
        let content = format!(
            "Modified {} files: {}",
            files_changed.len(),
            files_changed
                .iter()
                .map(|f| short_path(f))
                .collect::<Vec<_>>()
                .join(", ")
        );
        stats.extracted += upsert_knowledge(
            db, session_id, "architecture", subject, &content, 0.5,
        )?;
    }

    // 5. Extract from bash commands (build tool choices, test frameworks)
    let bash_turns: Vec<&search::TurnSearchResult> = turns
        .iter()
        .filter(|t| t.turn_type == "bash_cmd")
        .collect();

    for bash in &bash_turns {
        let content_lower = bash.content.to_lowercase();

        if content_lower.contains("$ cargo") {
            stats.extracted += upsert_knowledge(
                db, session_id, "convention", "build tool", "Uses Cargo (Rust)", 0.9,
            )?;
        } else if content_lower.contains("$ npm") || content_lower.contains("$ npx") {
            stats.extracted += upsert_knowledge(
                db, session_id, "convention", "build tool", "Uses npm", 0.9,
            )?;
        } else if content_lower.contains("$ yarn") {
            stats.extracted += upsert_knowledge(
                db, session_id, "convention", "build tool", "Uses yarn", 0.9,
            )?;
        } else if content_lower.contains("$ bun") {
            stats.extracted += upsert_knowledge(
                db, session_id, "convention", "build tool", "Uses bun", 0.9,
            )?;
        } else if content_lower.contains("$ pnpm") {
            stats.extracted += upsert_knowledge(
                db, session_id, "convention", "build tool", "Uses pnpm", 0.9,
            )?;
        } else if content_lower.contains("$ pip") || content_lower.contains("$ python") {
            stats.extracted += upsert_knowledge(
                db, session_id, "convention", "build tool", "Uses pip/Python", 0.9,
            )?;
        } else if content_lower.contains("$ go ") {
            stats.extracted += upsert_knowledge(
                db, session_id, "convention", "build tool", "Uses Go toolchain", 0.9,
            )?;
        }

        // Test framework detection
        if content_lower.contains("$ cargo test") || content_lower.contains("$ pytest")
            || content_lower.contains("$ npm test") || content_lower.contains("$ jest")
            || content_lower.contains("$ vitest")
        {
            let framework = if content_lower.contains("cargo test") {
                "cargo test"
            } else if content_lower.contains("pytest") {
                "pytest"
            } else if content_lower.contains("jest") {
                "jest"
            } else if content_lower.contains("vitest") {
                "vitest"
            } else {
                "npm test"
            };
            stats.extracted += upsert_knowledge(
                db,
                session_id,
                "convention",
                "test framework",
                &format!("Uses {}", framework),
                0.9,
            )?;
        }
    }

    Ok(stats)
}

/// Insert or update a knowledge entry.
/// Returns 1 if a new entry was created, 0 if an existing one was confirmed.
fn upsert_knowledge(
    db: &Db,
    session_id: &str,
    category: &str,
    subject: &str,
    content: &str,
    confidence: f64,
) -> Result<usize> {
    let conn = db.conn();

    // Check for existing entry with the same subject
    let existing: Option<(i64, f64, String)> = conn
        .query_row(
            "SELECT id, confidence, content FROM knowledge
             WHERE subject = ?1 AND category = ?2 AND superseded_by IS NULL
             ORDER BY created_at DESC LIMIT 1",
            params![subject, category],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .ok();

    match existing {
        Some((existing_id, existing_confidence, existing_content)) => {
            if contents_agree(&existing_content, content) {
                // Confirmed — boost confidence
                let new_confidence = (existing_confidence + 0.1).min(1.0);
                conn.execute(
                    "UPDATE knowledge SET confidence = ?1, last_confirmed = datetime('now')
                     WHERE id = ?2",
                    params![new_confidence, existing_id],
                )?;
                Ok(0)
            } else {
                // Contradiction — supersede the old entry
                conn.execute(
                    "UPDATE knowledge SET confidence = confidence * 0.5 WHERE id = ?1",
                    params![existing_id],
                )?;

                conn.execute(
                    "INSERT INTO knowledge (session_id, category, subject, content, confidence)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![session_id, category, subject, content, confidence],
                )?;

                let new_id = conn.last_insert_rowid();
                conn.execute(
                    "UPDATE knowledge SET superseded_by = ?1 WHERE id = ?2",
                    params![new_id, existing_id],
                )?;
                Ok(1)
            }
        }
        None => {
            // New entry
            conn.execute(
                "INSERT INTO knowledge (session_id, category, subject, content, confidence)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![session_id, category, subject, content, confidence],
            )?;
            Ok(1)
        }
    }
}

/// Simple heuristic: do two content strings agree?
/// They agree if they mention the same key terms.
fn contents_agree(a: &str, b: &str) -> bool {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    // Extract significant words (>3 chars, not common words)
    let common_words: std::collections::HashSet<&str> = [
        "the", "and", "for", "with", "this", "that", "from", "have", "will",
        "use", "uses", "used", "using", "into", "than", "over", "also",
    ]
    .iter()
    .copied()
    .collect();

    let a_words: std::collections::HashSet<&str> = a_lower
        .split_whitespace()
        .filter(|w| w.len() > 3 && !common_words.contains(w))
        .collect();

    let b_words: std::collections::HashSet<&str> = b_lower
        .split_whitespace()
        .filter(|w| w.len() > 3 && !common_words.contains(w))
        .collect();

    if a_words.is_empty() || b_words.is_empty() {
        return true; // Can't determine, assume agreement
    }

    let overlap = a_words.intersection(&b_words).count();
    let total = a_words.len().min(b_words.len());

    // If >40% of the shorter set overlaps, they agree
    overlap as f64 / total as f64 > 0.4
}

/// Extract a preference subject from text containing "always" or "never".
fn extract_preference_subject(text: &str) -> String {
    // Try to extract the phrase after "always" or "never"
    for keyword in &["always ", "never "] {
        if let Some(pos) = text.find(keyword) {
            let after = &text[pos + keyword.len()..];
            let end = after
                .find(|c: char| c == '.' || c == ',' || c == '!' || c == '\n')
                .unwrap_or_else(|| after.floor_char_boundary(after.len().min(50)));
            let phrase = after[..end].trim();
            if !phrase.is_empty() {
                return format!("user preference: {}", truncate(phrase, 60));
            }
        }
    }
    "user preference".to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let end = s.floor_char_boundary(max);
        format!("{}...", &s[..end])
    }
}

fn short_path(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("/")
}

#[derive(Debug, Default)]
pub struct DistillStats {
    pub extracted: usize,
}
