use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use seekdb_rs::{EmbeddingFunction, Embeddings, SeekDbError, ServerConfig};

/// Load ServerConfig from environment when `SEEKDB_INTEGRATION=1` is set.
/// Returns None and prints a SKIP message otherwise.
pub fn load_config_for_integration() -> Option<ServerConfig> {
    if std::env::var("SEEKDB_INTEGRATION").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set SEEKDB_INTEGRATION=1 and SERVER_* env vars to run integration tests");
        return None;
    }
    ServerConfig::from_env().ok()
}

/// Millisecond timestamp string used to make database/collection names unique.
pub fn ts_suffix() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    ts.to_string()
}

/// Dummy embedding function to satisfy type parameters; not used in these tests.
pub struct DummyEmbedding;

#[async_trait::async_trait]
impl EmbeddingFunction for DummyEmbedding {
    async fn embed_documents(&self, _docs: &[String]) -> Result<Embeddings, SeekDbError> {
        Err(SeekDbError::Embedding(
            "DummyEmbedding should not be called".into(),
        ))
    }

    fn dimension(&self) -> usize {
        3
    }
}

/// Simple embedding function that returns a constant vector of the given dimension.
pub struct ConstantEmbedding {
    pub value: f32,
    pub dim: usize,
}

#[async_trait::async_trait]
impl EmbeddingFunction for ConstantEmbedding {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings, SeekDbError> {
        let mut out = Vec::with_capacity(docs.len());
        for _ in docs {
            out.push(vec![self.value; self.dim]);
        }
        Ok(out)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

