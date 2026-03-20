use anyhow::Result;
use serde_json::json;
use std::path::Path;

use crate::db::Db;
use crate::db::search;
use crate::hooks::{self, HookInput};
use crate::llm::LlmConfig;

/// Handle PreToolUse hook.
///
/// For Task(subagent_type=Explore), builds a briefing from the project's
/// indexed symbols and knowledge. If Haiku is configured, the raw data is
/// synthesized into a focused report. Otherwise the raw data is injected
/// directly. Either way the subagent starts with answers instead of
/// instructions it would ignore.
pub fn handle(input: &HookInput) -> Result<()> {
    let tool_name = match &input.tool_name {
        Some(name) => name.as_str(),
        None => return Ok(()),
    };

    if tool_name != "Task" {
        return Ok(());
    }

    let tool_input = match &input.tool_input {
        Some(v) => v,
        None => return Ok(()),
    };

    let subagent_type = tool_input
        .get("subagent_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Log every PreToolUse invocation before any early returns
    let project_dir = hooks::project_dir(input);
    let log_db = Db::open(Path::new(&project_dir)).ok();

    if subagent_type != "Explore" {
        if let Some(db) = &log_db {
            hooks::log_hook(db, input, "PreToolUse", &format!("Task/{}: skipped", subagent_type));
        }
        return Ok(());
    }

    let original_prompt = tool_input
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if original_prompt.is_empty() {
        if let Some(db) = &log_db {
            hooks::log_hook(db, input, "PreToolUse", "Task/Explore: no prompt");
        }
        return Ok(());
    }

    match build_explore_briefing(&project_dir, original_prompt) {
        Ok(Some(briefing)) => {
            if let Some(db) = &log_db {
                let detail = format!("Task/Explore: briefing ({} bytes)", briefing.len());
                hooks::log_hook(db, input, "PreToolUse", &detail);
            }

            let augmented_prompt = format!(
                "{}\n\n---\nOriginal task:\n{}",
                briefing, original_prompt
            );

            let output = json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": {
                        "prompt": augmented_prompt
                    }
                }
            });
            println!("{}", serde_json::to_string(&output)?);
        }
        Ok(None) => {
            if let Some(db) = &log_db {
                hooks::log_hook(db, input, "PreToolUse", "Task/Explore: no data");
            }
        }
        Err(e) => {
            if let Some(db) = &log_db {
                hooks::log_hook(db, input, "PreToolUse", &format!("Task/Explore: error: {}", e));
            }
            eprintln!("[memory-rlm] Explore briefing failed: {}", e);
        }
    }

    Ok(())
}

/// Build a briefing for an Explore subagent.
///
/// 1. Extracts keywords from the prompt
/// 2. Queries the symbol index for matches
/// 3. Queries knowledge FTS for relevant entries
/// 4. If Haiku is configured, synthesizes a report; otherwise formats raw data
fn build_explore_briefing(project_dir: &str, prompt: &str) -> Result<Option<String>> {
    let db = Db::open(Path::new(project_dir))?;
    let conn = db.conn();

    // 1. Extract search keywords from the prompt
    let keywords = extract_keywords(prompt);
    if keywords.is_empty() {
        return Ok(None);
    }

    // 2. Query symbols by keywords
    let symbols = search::search_symbols_by_keywords(&conn, &keywords, 50)?;

    // 3. Query knowledge — do per-keyword searches with OR semantics
    let mut knowledge = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();
    for kw in keywords.iter().take(5) {
        if let Ok(results) = search::search_knowledge(&conn, kw, 3, None) {
            for r in results {
                if seen_ids.insert(r.id) {
                    knowledge.push(r);
                }
            }
        }
    }
    knowledge.truncate(8);

    // Release DB lock before any network call
    drop(conn);

    if symbols.is_empty() && knowledge.is_empty() {
        return Ok(None);
    }

    // 4. Format raw data
    let data_section = format_raw_data(&symbols, &knowledge, project_dir);

    // 5. Try Haiku synthesis, fall back to raw data
    if let Some(llm) = LlmConfig::from_env() {
        match synthesize_briefing(&llm, prompt, &data_section) {
            Ok(report) => {
                return Ok(Some(format!(
                    "## Pre-computed Briefing\n\
                     This was prepared from the project's indexed symbols and knowledge.\n\
                     Use it as your starting point — only explore files for gaps.\n\n\
                     {}",
                    report
                )));
            }
            Err(e) => {
                eprintln!("[memory-rlm] Haiku synthesis failed, using raw data: {}", e);
            }
        }
    }

    // Fallback: inject raw data directly
    Ok(Some(format!(
        "## Pre-computed Index Data\n\
         The following symbols and knowledge were found in the project index.\n\
         Use this as your starting point.\n\n\
         {}",
        data_section
    )))
}

/// Format symbol matches and knowledge into a readable data section.
fn format_raw_data(
    symbols: &[search::SymbolMatch],
    knowledge: &[search::KnowledgeSearchResult],
    project_dir: &str,
) -> String {
    let prefix = project_dir.replace('\\', "/");
    let prefix = prefix.trim_end_matches('/');

    let mut out = String::new();

    if !symbols.is_empty() {
        out.push_str("### Matching Symbols\n");
        for sym in symbols {
            let rel_path = make_relative(&sym.file_path, prefix);
            let parent = sym
                .parent_name
                .as_deref()
                .map(|p| format!("{}::", p))
                .unwrap_or_default();

            out.push_str(&format!(
                "- {} {}{} — {}:{}-{}\n",
                sym.kind, parent, sym.name, rel_path, sym.start_line, sym.end_line,
            ));
            if let Some(sig) = &sym.signature {
                if !sig.is_empty() {
                    out.push_str(&format!("  `{}`\n", truncate(sig, 120)));
                }
            }
            if let Some(doc) = &sym.doc_comment {
                if !doc.is_empty() {
                    out.push_str(&format!("  {}\n", truncate(doc, 120)));
                }
            }
        }
    }

    if !knowledge.is_empty() {
        out.push_str("\n### Relevant Knowledge\n");
        for k in knowledge {
            out.push_str(&format!(
                "- [{}] **{}**: {}\n",
                k.category,
                k.subject,
                truncate(&k.content, 200)
            ));
        }
    }

    out
}

/// Call Haiku to synthesize a briefing from the raw data.
fn synthesize_briefing(llm: &LlmConfig, prompt: &str, data: &str) -> Result<String> {
    let system = "\
You are a codebase analyst preparing a briefing for a code exploration agent.
Given the agent's task and indexed project data, produce a concise report:
1. Group related symbols and explain how they connect
2. Note file locations using file:line format
3. Include any relevant project knowledge or decisions
4. Flag gaps — what the agent should still investigate via file reads

Be concise and factual. No preamble. Use markdown formatting.";

    let user_msg = format!(
        "## Agent's Task\n{}\n\n## Indexed Data\n{}\n\nProduce a briefing report.",
        prompt, data
    );

    llm.complete(system, &user_msg)
}

/// Extract meaningful search keywords from a prompt.
///
/// Filters out English stop words and common code-exploration boilerplate
/// (e.g., "find", "function", "file") to leave domain-specific terms.
fn extract_keywords(prompt: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        // English
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "must",
        "it", "its", "this", "that", "these", "those", "you", "we", "they",
        "me", "him", "her", "them", "my", "your", "his", "our", "their",
        "what", "which", "who", "when", "where", "how", "not", "no", "nor",
        "if", "then", "else", "so", "than", "too", "very", "just", "about",
        "up", "out", "all", "any", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "only", "into", "also", "well",
        // Code exploration boilerplate
        "find", "search", "look", "show", "list", "get", "check", "see",
        "file", "files", "code", "codebase", "project", "directory",
        "function", "functions", "class", "classes", "method", "methods",
        "struct", "structs", "implementation", "implementations",
        "module", "modules", "related", "relevant", "existing", "current",
        "using", "used", "understand", "understanding", "want", "need",
        "return", "returns", "here", "there", "between", "within",
        "across", "through", "whether", "including", "like", "use",
    ];

    let stop_set: std::collections::HashSet<&str> = STOP_WORDS.iter().copied().collect();

    prompt
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() > 2)
        .map(|w| w.to_lowercase())
        .filter(|w| !stop_set.contains(w.as_str()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect()
}

fn make_relative(path: &str, prefix: &str) -> String {
    let normalized = path.replace('\\', "/");
    normalized
        .strip_prefix(prefix)
        .unwrap_or(&normalized)
        .trim_start_matches('/')
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let end = s.floor_char_boundary(max);
        format!("{}...", &s[..end])
    }
}
