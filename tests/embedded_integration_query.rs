//! Integration tests for embedded collection query/get operations.
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
    query_execute_and_fetch().await?;
    query_filter_where().await?;
    Ok(())
}

/// Execute + fetch_all (mirrors server collection_query_and_filters style).
#[cfg(feature = "embedded")]
async fn query_execute_and_fetch() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    client.execute(
        "CREATE TABLE IF NOT EXISTS test_vectors (_id VARBINARY(512) PRIMARY KEY, document TEXT, embedding VECTOR(3), metadata JSON)",
    ).await?;
    client.execute(
        "INSERT INTO test_vectors (_id, document, embedding, metadata) VALUES (X'616263', 'test doc 1', '[0,0,0]', '{}')",
    ).await?;
    let rows = client.fetch_all("SELECT _id, document FROM test_vectors").await?;
    assert_eq!(rows.len(), 1);
    client.execute("DROP TABLE IF EXISTS test_vectors").await?;
    Ok(())
}

/// SQL WHERE filter (mirrors server collection_query_and_filters).
#[cfg(feature = "embedded")]
async fn query_filter_where() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    client.execute("CREATE TABLE IF NOT EXISTS test_filter (id INT PRIMARY KEY, score INT)").await?;
    client.execute("INSERT INTO test_filter (id, score) VALUES (1, 10), (2, 20)").await?;
    let rows = client.fetch_all("SELECT * FROM test_filter WHERE score > 15").await?;
    assert_eq!(rows.len(), 1);
    client.execute("DROP TABLE IF EXISTS test_filter").await?;
    Ok(())
}
