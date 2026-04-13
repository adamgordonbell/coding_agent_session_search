use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use frankensearch::{ModelCategory, ModelTier};

use super::daemon_client::DaemonClient;
use super::embedder::{Embedder, EmbedderError, EmbedderResult};

pub struct DaemonMiniLmEmbedder {
    daemon: Arc<dyn DaemonClient>,
    request_counter: AtomicU64,
}

impl DaemonMiniLmEmbedder {
    pub fn new(daemon: Arc<dyn DaemonClient>) -> Self {
        Self {
            daemon,
            request_counter: AtomicU64::new(1),
        }
    }

    fn next_request_id(&self) -> String {
        format!(
            "daemon-embed-{}",
            self.request_counter.fetch_add(1, Ordering::Relaxed)
        )
    }
}

impl Embedder for DaemonMiniLmEmbedder {
    fn embed_sync(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbedderError::InvalidConfig {
                field: "input_text".to_string(),
                value: "(empty)".to_string(),
                reason: "empty text".to_string(),
            });
        }

        self.daemon
            .embed(text, &self.next_request_id())
            .map_err(|err| EmbedderError::EmbeddingFailed {
                model: self.id().to_string(),
                source: Box::new(std::io::Error::other(err.to_string())),
            })
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        for text in texts {
            if text.is_empty() {
                return Err(EmbedderError::InvalidConfig {
                    field: "input_text".to_string(),
                    value: "(empty)".to_string(),
                    reason: "empty text in batch".to_string(),
                });
            }
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        self.daemon
            .embed_batch(texts, &self.next_request_id())
            .map_err(|err| EmbedderError::EmbeddingFailed {
                model: self.id().to_string(),
                source: Box::new(std::io::Error::other(err.to_string())),
            })
    }

    fn dimension(&self) -> usize {
        384
    }

    fn id(&self) -> &str {
        "minilm-384"
    }

    fn model_name(&self) -> &str {
        "all-minilm-l6-v2-daemon"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }

    fn tier(&self) -> ModelTier {
        ModelTier::Quality
    }
}
