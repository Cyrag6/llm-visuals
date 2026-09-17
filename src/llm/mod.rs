use serde::Deserialize;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::pipeline::AttentionWeight;

/// The python.org installer on Windows provides `python`, not `python3`.
const PYTHON: &str = if cfg!(windows) { "python" } else { "python3" };

/// Events emitted by the Python HF bridge (JSONL on stdout)
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BridgeEvent {
    Status {
        message: String,
    },
    ModelInfo {
        num_layers: Option<usize>,
        num_heads: Option<usize>,
        #[serde(default)]
        ctx_max: Option<usize>,
        model: String,
    },
    Attention {
        layer: usize,
        head: usize,
        query_pos: usize,
        #[allow(dead_code)]
        key_pos: i64,
        weight: f32,
    },
    Token {
        token_index: usize,
        #[allow(dead_code)]
        token_id: u32,
        text: String,
    },
    Done {
        tokens_generated: usize,
    },
    Error {
        message: String,
    },
}

/// High-level events for the UI / aggregator
#[derive(Debug, Clone)]
pub enum LlmEvent {
    Status(String),
    #[allow(dead_code)]
    ModelInfo {
        num_layers: usize,
        num_heads: usize,
        ctx_max: usize,
        model: String,
    },
    Attention(AttentionWeight),
    #[allow(dead_code)]
    Token {
        index: usize,
        text: String,
    },
    Done {
        tokens_generated: usize,
    },
    Error(String),
}

pub struct PythonBridge {
    #[allow(dead_code)]
    child: Child,
}

impl PythonBridge {
    /// Spawn the Python bridge and return a receiver of LLM events.
    pub async fn spawn(
        model: &str,
        prompt: &str,
        max_tokens: usize,
        bridge_script: PathBuf,
    ) -> Result<(Self, mpsc::Receiver<LlmEvent>), String> {
        if !bridge_script.exists() {
            return Err(format!(
                "Python bridge script not found: {}",
                bridge_script.display()
            ));
        }

        let mut child = Command::new(PYTHON)
            .arg(&bridge_script)
            .arg("--model")
            .arg(model)
            .arg("--prompt")
            .arg(prompt)
            .arg("--max-tokens")
            .arg(max_tokens.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn {PYTHON}: {e}"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture python stdout".to_string())?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(_)) = reader.next_line().await {}
            });
        }

        let (tx, rx) = mpsc::channel::<LlmEvent>(8192);

        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<BridgeEvent>(&line) {
                    Ok(event) => {
                        let mapped = match event {
                            BridgeEvent::Status { message } => LlmEvent::Status(message),
                            BridgeEvent::ModelInfo {
                                num_layers,
                                num_heads,
                                ctx_max,
                                model,
                            } => LlmEvent::ModelInfo {
                                num_layers: num_layers.unwrap_or(0),
                                num_heads: num_heads.unwrap_or(0),
                                ctx_max: ctx_max.unwrap_or(8192),
                                model,
                            },
                            BridgeEvent::Attention {
                                layer,
                                head,
                                query_pos,
                                key_pos: _,
                                weight,
                            } => LlmEvent::Attention(AttentionWeight {
                                layer,
                                head,
                                query_pos,
                                key_pos: 0,
                                weight,
                            }),
                            BridgeEvent::Token {
                                token_index,
                                token_id: _,
                                text,
                            } => LlmEvent::Token {
                                index: token_index,
                                text,
                            },
                            BridgeEvent::Done { tokens_generated } => {
                                LlmEvent::Done { tokens_generated }
                            }
                            BridgeEvent::Error { message } => LlmEvent::Error(message),
                        };
                        if tx.send(mapped).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(LlmEvent::Error(format!(
                                "Bad JSON from bridge: {e} | line={line}"
                            )))
                            .await;
                    }
                }
            }
        });

        Ok((Self { child }, rx))
    }

    #[allow(dead_code)]
    pub async fn wait(mut self) -> Result<(), String> {
        let status = self
            .child
            .wait()
            .await
            .map_err(|e| format!("Failed waiting for python: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("Python bridge exited with {status}"))
        }
    }
}
