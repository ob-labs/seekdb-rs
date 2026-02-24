#![cfg(feature = "embedded")]
#![allow(dead_code)] // Helpers are used by different test binaries; not all use every symbol.
//! Shared helpers for embedded integration tests.
//! Open DB once on main thread, then run async tests via run_embedded_tests(run_tests).

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use seekdb_rs::{EmbeddedConfig, EmbeddedDatabase, EmbeddingFunction, Embeddings, SeekDbError};


/// Single database directory for all embedded integration tests.
/// Uses `tests/seekdb.db`, normalized to absolute path so open and connect use the same path regardless of cwd.
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

/// Unified entry: open DB once on main thread with shared path, then run async tests.
/// Call from each test's `main()` as `common::run_embedded_tests(run_tests)`.
/// All test cases use the same directory via `shared_db_dir()`; do not call `open` again in tests.
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
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    match rt.block_on(run()) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("{:?}", e);
            std::process::exit(1);
        }
    }
}

/// When false, tests that call `EmbeddedDatabase::open()` should skip (return Ok(())),
/// to avoid SIGSEGV when open runs on a non-main thread (see docs/debugging_embedded.md).
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
