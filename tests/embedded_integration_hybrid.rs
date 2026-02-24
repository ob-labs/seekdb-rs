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
    embedded_hybrid_search_placeholder().await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_hybrid_search_placeholder() -> Result<()> {
    let db_dir = shared_db_dir();
    let _client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    Ok(())
}
