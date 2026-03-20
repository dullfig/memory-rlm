use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// LLM provider configuration.
///
/// Loaded from config file first, then environment variables as fallback.
///
/// Config file locations (checked in order):
///   1. <project_dir>/.claude/memory-rlm.toml   (project-level)
///   2. ~/.config/memory-rlm/config.toml         (global, Linux/macOS)
///      %APPDATA%\memory-rlm\config.toml         (global, Windows)
///   3. Environment variables (CONTEXTMEM_LLM_*)
///
/// Config format:
///   [llm]
///   provider = "auto"            # "auto", "local", "anthropic", "openai", "ollama"
///   api_key = "sk-ant-..."
///   model = "claude-haiku-4-5-20251001"
///   base_url = "https://api.anthropic.com"
///   local_model_path = "/path/to/model.gguf"       # optional, default auto-downloads
///   local_tokenizer_path = "/path/to/tokenizer.json"
///   local_speed_threshold = 15.0                   # min tok/s to use local
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub provider: Provider,
    pub api_key: Option<String>,
    pub model: String,
    pub base_url: String,
    /// Path to a GGUF model file (optional, auto-downloads default model if unset).
    pub local_model_path: Option<String>,
    /// Path to a tokenizer.json file (optional, auto-downloads if unset).
    pub local_tokenizer_path: Option<String>,
    /// Minimum tokens/second to consider local inference viable (default: 15.0).
    pub local_speed_threshold: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    Anthropic,
    OpenAICompat,
    /// Force local model inference via Candle.
    Local,
    /// Auto-detect: use local if benchmarked and fast enough, else fall back to API.
    Auto,
}

/// The [llm] section of the config file.
#[derive(Debug, Deserialize, Default)]
struct FileConfig {
    #[serde(default)]
    llm: LlmFileConfig,
}

#[derive(Debug, Deserialize, Default)]
struct LlmFileConfig {
    provider: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    local_model_path: Option<String>,
    local_tokenizer_path: Option<String>,
    local_speed_threshold: Option<f64>,
}

impl LlmConfig {
    /// Load configuration from config file, falling back to environment variables.
    /// Returns None if no LLM is configured.
    pub fn from_env() -> Option<Self> {
        let file_cfg = load_config_file().unwrap_or_default();

        let api_key = file_cfg.llm.api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .or_else(|| std::env::var("CONTEXTMEM_LLM_API_KEY").ok());

        let provider_str = file_cfg.llm.provider
            .or_else(|| std::env::var("CONTEXTMEM_LLM_PROVIDER").ok())
            .unwrap_or_else(|| "auto".to_string());

        let provider = match provider_str.to_lowercase().as_str() {
            "openai" | "ollama" | "openrouter" => Provider::OpenAICompat,
            "anthropic" => Provider::Anthropic,
            "local" => Provider::Local,
            _ => Provider::Auto,
        };

        let (default_model, default_url) = match provider {
            Provider::Anthropic | Provider::Auto => (
                "claude-haiku-4-5-20251001".to_string(),
                "https://api.anthropic.com".to_string(),
            ),
            Provider::OpenAICompat => (
                "llama3".to_string(),
                "http://localhost:11434".to_string(),
            ),
            Provider::Local => (
                String::new(),
                String::new(),
            ),
        };

        let model = file_cfg.llm.model
            .or_else(|| std::env::var("CONTEXTMEM_LLM_MODEL").ok())
            .unwrap_or(default_model);

        let base_url = file_cfg.llm.base_url
            .or_else(|| std::env::var("CONTEXTMEM_LLM_BASE_URL").ok())
            .unwrap_or(default_url);

        let local_model_path = file_cfg.llm.local_model_path
            .or_else(|| std::env::var("CONTEXTMEM_LOCAL_MODEL_PATH").ok());
        let local_tokenizer_path = file_cfg.llm.local_tokenizer_path
            .or_else(|| std::env::var("CONTEXTMEM_LOCAL_TOKENIZER_PATH").ok());
        let local_speed_threshold = file_cfg.llm.local_speed_threshold
            .unwrap_or(15.0);

        // For Anthropic-only, require an API key
        // For Auto, API key is optional (will try local first)
        // For Local, no API key needed
        // For OpenAI-compat (Ollama), API key is optional
        if provider == Provider::Anthropic && api_key.is_none() {
            eprintln!(
                "[memory-rlm] No LLM API key found. Set ANTHROPIC_API_KEY or run: memory-rlm config set api-key <key>"
            );
            return None;
        }

        Some(Self {
            provider,
            api_key,
            model,
            base_url,
            local_model_path,
            local_tokenizer_path,
            local_speed_threshold,
        })
    }

    /// Send a prompt to the configured LLM and return the response text.
    pub fn complete(&self, system: &str, user_message: &str) -> Result<String> {
        match self.provider {
            Provider::Anthropic => self.complete_anthropic(system, user_message),
            Provider::OpenAICompat => self.complete_openai(system, user_message),
            #[cfg(feature = "local-inference")]
            Provider::Local => self.do_complete_local(system, user_message),
            #[cfg(feature = "local-inference")]
            Provider::Auto => self.complete_auto(system, user_message),
            #[cfg(not(feature = "local-inference"))]
            Provider::Local | Provider::Auto => {
                Err(anyhow!("Local inference not available. Rebuild with --features local-inference"))
            }
        }
    }

    /// Call the Anthropic Messages API.
    fn complete_anthropic(&self, system: &str, user_message: &str) -> Result<String> {
        let api_key = self.api_key.as_deref()
            .ok_or_else(|| anyhow!("api_key required for Anthropic provider"))?;

        let client = reqwest::blocking::Client::new();
        let url = format!("{}/v1/messages", self.base_url);

        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: 2048,
            system,
            messages: vec![AnthropicMessage {
                role: "user",
                content: user_message,
            }],
        };

        let resp = client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .timeout(std::time::Duration::from_secs(30))
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(anyhow!("Anthropic API error {}: {}", status, body));
        }

        let resp: AnthropicResponse = resp.json()?;
        resp.content
            .into_iter()
            .find(|b| b.content_type == "text")
            .map(|b| b.text)
            .ok_or_else(|| anyhow!("No text content in Anthropic response"))
    }

    /// Call an OpenAI-compatible Chat Completions API (works with Ollama, OpenRouter, etc.)
    fn complete_openai(&self, system: &str, user_message: &str) -> Result<String> {
        let client = reqwest::blocking::Client::new();

        // Ollama uses /api/chat, but most OpenAI-compat use /v1/chat/completions
        let url = if self.base_url.contains("localhost:11434")
            || self.base_url.contains("127.0.0.1:11434")
        {
            format!("{}/api/chat", self.base_url)
        } else {
            format!("{}/v1/chat/completions", self.base_url)
        };

        let body = OpenAIRequest {
            model: &self.model,
            messages: vec![
                OpenAIMessage {
                    role: "system",
                    content: system,
                },
                OpenAIMessage {
                    role: "user",
                    content: user_message,
                },
            ],
            temperature: 0.3,
        };

        let mut req = client
            .post(&url)
            .header("content-type", "application/json")
            .timeout(std::time::Duration::from_secs(60));

        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let resp = req.json(&body).send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(anyhow!("OpenAI-compat API error {}: {}", status, body));
        }

        let resp: OpenAIResponse = resp.json()?;
        resp.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow!("No choices in OpenAI-compat response"))
    }

    /// Auto-detect: check cached benchmark, use local if fast enough, else API.
    #[cfg(feature = "local-inference")]
    fn complete_auto(&self, system: &str, user_message: &str) -> Result<String> {
        let assessment = crate::compute::assess_compute(self.local_speed_threshold);

        if assessment.use_local {
            eprintln!(
                "[memory-rlm] Auto: local inference ({:.1} tok/s)",
                assessment.tokens_per_second.unwrap_or(0.0)
            );
            match self.do_complete_local(system, user_message) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("[memory-rlm] Local inference failed, falling back to API: {}", e);
                }
            }
        } else {
            eprintln!("[memory-rlm] Auto: using API ({})", assessment.reason);
        }

        // Fallback to Anthropic
        if self.api_key.is_some() {
            self.complete_anthropic(system, user_message)
        } else {
            Err(anyhow!(
                "Local inference not viable ({}) and no API key configured",
                assessment.reason
            ))
        }
    }

    /// Run inference using a local Candle model.
    #[cfg(feature = "local-inference")]
    fn do_complete_local(&self, system: &str, user_message: &str) -> Result<String> {
        use std::path::PathBuf;

        if let (Some(model_path), Some(tokenizer_path)) =
            (&self.local_model_path, &self.local_tokenizer_path)
        {
            let mut model = crate::local_model::LocalModel::load(
                &PathBuf::from(model_path),
                &PathBuf::from(tokenizer_path),
            )?;
            model.complete(system, user_message, 2048)
        } else {
            crate::local_model::complete_local(system, user_message, 2048)
        }
    }
}

/// Load config from file. Checks project-level, then global.
fn load_config_file() -> Option<FileConfig> {
    let candidates = config_file_paths();
    for path in candidates {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            match toml::from_str(&contents) {
                Ok(cfg) => return Some(cfg),
                Err(e) => {
                    eprintln!("[memory-rlm] Warning: failed to parse {}: {}", path.display(), e);
                }
            }
        }
    }
    None
}

/// Return candidate config file paths in priority order.
fn config_file_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Project-level: <cwd>/.claude/memory-rlm.toml
    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join(".claude").join("memory-rlm.toml"));
    }

    // 2. Global config
    if cfg!(windows) {
        if let Ok(appdata) = std::env::var("APPDATA") {
            paths.push(PathBuf::from(appdata).join("memory-rlm").join("config.toml"));
        }
    } else {
        if let Ok(home) = std::env::var("HOME") {
            paths.push(PathBuf::from(home).join(".config").join("memory-rlm").join("config.toml"));
        }
    }

    paths
}

/// Return the path to the global config file.
pub fn global_config_path() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var("APPDATA")
            .ok()
            .map(|d| PathBuf::from(d).join("memory-rlm").join("config.toml"))
    } else {
        std::env::var("HOME")
            .ok()
            .map(|d| PathBuf::from(d).join(".config").join("memory-rlm").join("config.toml"))
    }
}

/// Write a key-value pair into a section of the global config TOML.
/// Creates the file and parent directories if needed. Merges with existing content.
pub fn write_global_config(section: &str, key: &str, value: toml::Value) -> Result<()> {
    let path = global_config_path()
        .ok_or_else(|| anyhow!("Cannot determine global config path"))?;

    // Load existing file or start fresh
    let mut doc: toml::Table = if path.exists() {
        let contents = std::fs::read_to_string(&path)?;
        contents.parse().unwrap_or_default()
    } else {
        toml::Table::new()
    };

    // Ensure [section] table exists
    let sect = doc
        .entry(section)
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let sect_table = sect
        .as_table_mut()
        .ok_or_else(|| anyhow!("'{}' key in config is not a table", section))?;

    sect_table.insert(key.to_string(), value);

    // Write back
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, doc.to_string())?;

    Ok(())
}

// --- Anthropic API types ---

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

// --- OpenAI-compatible API types ---

#[derive(Serialize)]
struct OpenAIRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAIMessage<'a>>,
    temperature: f32,
}

#[derive(Serialize)]
struct OpenAIMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIChoiceMessage,
}

#[derive(Deserialize)]
struct OpenAIChoiceMessage {
    content: String,
}
