//! Integration tests that mirror the README examples for embedded mode.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::{Client, EmbeddedClient, EmbeddedConfig, SeekDbClient};
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
    embedded_readme_basic_example().await?;
    embedded_readme_from_env_example().await?;
    embedded_seekdb_client_trait().await?;
    embedded_generic_client_usage().await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_readme_basic_example() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = EmbeddedClient::builder()
        .db_dir(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    client.execute("SELECT 1").await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_readme_from_env_example() -> Result<()> {
    let db_dir = shared_db_dir();
    unsafe {
        std::env::set_var("SEEKDB_EMBEDDED_INTEGRATION", "1");
        std::env::set_var("EMBEDDED_DB_DIR", db_dir.to_string_lossy().as_ref());
        std::env::set_var("EMBEDDED_DATABASE", "test");
    }
    let config = EmbeddedConfig::from_env()?;
    let client = EmbeddedClient::builder()
        .db_dir(config.db_dir.as_str())
        .database(config.database.as_str())
        .autocommit(config.autocommit)
        .skip_open(true)
        .build()
        .await?;
    client.execute("SELECT 1").await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_seekdb_client_trait() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    assert_eq!(client.database(), "test");
    let _ = client.list_collections().await?;
    let _ = client.count_collection().await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_generic_client_usage() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = EmbeddedClient::builder()
        .db_dir(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    async fn use_any_client<C: SeekDbClient>(client: &C) -> Result<()> {
        client.execute("SELECT 1").await?;
        let _ = client.list_collections().await?;
        Ok(())
    }
    use_any_client(&client).await?;
    Ok(())
}
