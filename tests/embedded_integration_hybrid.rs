//! Integration tests for embedded hybrid_search operations.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::Client;
#[cfg(feature = "embedded")]
#[path = "embedded/common.rs"]
mod common;
#[cfg(feature = "embedded")]
use common::shared_db_dir;

#[cfg(feature = "embedded")]
fn main() {
    common::run_embedded_tests(run_tests);
}

#[cfg(feature = "embedded")]
async fn run_tests() -> Result<()> {
    collection_hybrid_search_basic().await?;
    Ok(())
}

/// Placeholder for hybrid search (mirrors server collection_hybrid_search_basic).
#[cfg(feature = "embedded")]
async fn collection_hybrid_search_basic() -> Result<()> {
    let db_dir = shared_db_dir();
    let _client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    Ok(())
}
