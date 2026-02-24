//! Integration tests for embedded collection DML operations.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::{AdminApi, Client, SeekDbError};
#[cfg(feature = "embedded")]
#[path = "embedded/common.rs"]
mod common;
#[cfg(feature = "embedded")]
use common::{shared_db_dir, DummyEmbedding, ts_suffix};

#[cfg(feature = "embedded")]
fn main() {
    common::run_embedded_tests(run_tests);
}

#[cfg(feature = "embedded")]
async fn run_tests() -> Result<()> {
    embedded_collection_create_without_hnsw_config_errors().await?;
    embedded_collection_list_and_has().await?;
    embedded_collection_get_or_create().await?;
    embedded_database_operations().await?;
    embedded_sql_execution().await?;
    embedded_sql_error_handling().await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_collection_create_without_hnsw_config_errors() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    let name = format!("no_cfg_coll_{}", ts_suffix());
    let res = client
        .create_collection::<DummyEmbedding>(&name, None, None::<DummyEmbedding>)
        .await;
    match res {
        Err(SeekDbError::Config(msg)) => assert!(msg.contains("HnswConfig must be provided")),
        Ok(_) => panic!("expected SeekDbError::Config"),
        Err(e) => panic!("expected Config, got {:?}", e),
    }
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_collection_list_and_has() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    let coll_name = format!("test_coll_{}", ts_suffix());
    let _ = client.list_collections().await?;
    assert!(!client.has_collection(&coll_name).await?);
    let _ = client.count_collection().await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_collection_get_or_create() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    let coll_name = format!("get_or_create_{}", ts_suffix());
    let hnsw = seekdb_rs::HnswConfig {
        dimension: 3,
        distance: seekdb_rs::DistanceMetric::Cosine,
    };
    let coll = client
        .get_or_create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
    assert_eq!(coll.name(), &coll_name);
    assert_eq!(coll.dimension(), 3);
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_database_operations() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    let db_name = format!("test_db_{}", ts_suffix());
    client.create_database(&db_name, None).await?;
    let db = client.get_database(&db_name, None).await?;
    assert_eq!(db.name, db_name);
    client.delete_database(&db_name, None).await?;
    let list = client.list_databases(None, None, None).await?;
    assert!(!list.iter().any(|d| d.name == db_name));
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_sql_execution() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    client.execute("CREATE TABLE IF NOT EXISTS test_table (id INT, name VARCHAR(100))").await?;
    client.execute("INSERT INTO test_table (id, name) VALUES (1, 'test1')").await?;
    let rows = client.fetch_all("SELECT id, name FROM test_table ORDER BY id").await?;
    assert_eq!(rows.len(), 1);
    client.execute("DROP TABLE IF EXISTS test_table").await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn embedded_sql_error_handling() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .skip_open(true)
        .build()
        .await?;
    assert!(client.execute("SELECT * FROM non_existent_table").await.is_err());
    Ok(())
}
