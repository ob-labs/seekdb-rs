//! Integration tests for embedded mode with DefaultEmbedding.
#![cfg_attr(not(all(feature = "embedded", feature = "embedding")), allow(dead_code))]

#[cfg(all(feature = "embedded", feature = "embedding"))]
#[path = "embedded/common.rs"]
mod common;

#[cfg(not(all(feature = "embedded", feature = "embedding")))]
fn main() {}

#[cfg(all(feature = "embedded", feature = "embedding"))]
mod run {
    use anyhow::Result;
    use seekdb_rs::{Client, DefaultEmbedding, EmbeddingFunction};

    use crate::common::{run_embedded_tests, shared_db_dir};

    pub fn main() {
        run_embedded_tests(run_tests);
    }

    async fn run_tests() -> Result<()> {
        embedded_default_embedding_placeholder().await?;
        Ok(())
    }

    async fn embedded_default_embedding_placeholder() -> Result<()> {
        let db_dir = shared_db_dir();
        let _client = Client::builder()
            .path(db_dir.to_string_lossy().as_ref())
            .database("test")
            .skip_open(true)
            .build()
            .await?;
        let ef = DefaultEmbedding::new()?;
        assert!(ef.dimension() > 0);
        Ok(())
    }
}

#[cfg(all(feature = "embedded", feature = "embedding"))]
fn main() {
    run::main();
}
