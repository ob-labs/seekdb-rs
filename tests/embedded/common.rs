#![cfg(feature = "embedded")]
#![allow(dead_code)] // Helpers are used by different test binaries; not all use every symbol.
//! Shared helpers for embedded integration tests.
//! run_embedded_tests: main thread open() then block_on(sentinel + tests) then close(). Sentinel keeps one connection so the last client drop does not trigger close between cases; close-then-open in C library can hang (re-init after soft close).

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use seekdb_rs::{EmbeddedConfig, EmbeddedClient, EmbeddedDatabase, EmbeddingFunction, Embeddings, SeekDbError};

/// Single database directory for all embedded integration tests.
pub fn shared_db_dir() -> PathBuf {
    let base = PathBuf::from("tests/seekdb.db");
    if base.is_absolute() {
        base
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(base)
    }
}

/// Main thread: open() once, hold a sentinel connection, run tests, then close(). Sentinel avoids close between cases so we never do close-then-open (re-init can hang).
pub fn run_embedded_tests<Fut>(run: fn() -> Fut)
where
    Fut: std::future::Future<Output = Result<()>>,
{
    let init_dir = shared_db_dir();
    if let Err(e) = std::fs::create_dir_all(&init_dir) {
        eprintln!("create_dir_all: {:?}", e);
        std::process::exit(1);
    }
    if let Err(e) = EmbeddedDatabase::open(&init_dir) {
        eprintln!("{:?}", e);
        std::process::exit(1);
    }
    let dir_str = init_dir.to_string_lossy().to_string();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result = rt.block_on(async {
        let _sentinel = EmbeddedClient::builder()
            .db_dir(&dir_str)
            .database("test")
            .build()
            .await?;
        run().await
    });
    EmbeddedDatabase::close();
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("{:?}", e);
            std::process::exit(1);
        }
    }
}

/// When false, tests that call `EmbeddedDatabase::open()` should skip (return Ok(())),
/// since open must run on the main thread.
pub fn skip_if_no_integration() -> bool {
    std::env::var("SEEKDB_EMBEDDED_INTEGRATION").ok().as_deref() != Some("1")
}

/// Load EmbeddedConfig from environment when `SEEKDB_EMBEDDED_INTEGRATION=1` is set.
/// Returns None and prints a SKIP message otherwise.
pub fn load_config_for_integration() -> Option<EmbeddedConfig> {
    if std::env::var("SEEKDB_EMBEDDED_INTEGRATION").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set SEEKDB_EMBEDDED_INTEGRATION=1 and EMBEDDED_* env vars to run embedded integration tests");
        return None;
    }
    EmbeddedConfig::from_env().ok()
}

/// Alias for shared_db_dir(); kept for compatibility.
pub fn temp_db_dir() -> PathBuf {
    shared_db_dir()
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
