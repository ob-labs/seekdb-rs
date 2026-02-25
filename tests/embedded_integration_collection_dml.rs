//! Integration tests for embedded collection DML operations.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::{AdminApi, Client, Embedding, SeekDbError};
#[cfg(feature = "embedded")]
use serde_json::json;
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
    collection_create_without_hnsw_config_errors().await?;
    collection_list_and_has().await?;
    collection_get_or_create().await?;
    collection_get_or_create_and_legacy_dml().await?;
    database_operations().await?;
    sql_execution().await?;
    sql_error_handling().await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_create_without_hnsw_config_errors() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
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
async fn collection_list_and_has() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("test_coll_{}", ts_suffix());
    let _ = client.list_collections().await?;
    assert!(!client.has_collection(&coll_name).await?);
    let _ = client.count_collection().await?;
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_get_or_create() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
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

/// README-style flow: get_or_create_collection, list/get/has/count_collection, legacy add/update/get/delete/count (mirrors server).
#[cfg(feature = "embedded")]
async fn collection_get_or_create_and_legacy_dml() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("readme_coll_{}", ts_suffix());
    let hnsw = seekdb_rs::HnswConfig {
        dimension: 3,
        distance: seekdb_rs::DistanceMetric::Cosine,
    };
    if client.has_collection(&coll_name).await? {
        client.delete_collection(&coll_name).await?;
    }
    let coll = client
        .get_or_create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    let _ = client.list_collections().await?;
    let _ = client
        .get_collection::<DummyEmbedding>(&coll_name, None::<DummyEmbedding>)
        .await?;
    assert!(client.has_collection(&coll_name).await?);
    let _ = client.count_collection().await?;

    let ids = vec!["item1".to_string(), "item2".to_string()];
    let embeddings: Vec<Embedding> = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let documents = vec!["Document 1".to_string(), "Document 2".to_string()];
    let metadatas = vec![json!({"category": "AI"}), json!({"category": "ML"})];
    coll.add(&ids, Some(&embeddings), Some(&metadatas), Some(&documents))
        .await?;
    coll.update(
        &["item1".to_string()],
        Some(&[vec![0.7, 0.8, 0.9]]),
        Some(&[json!({"category": "AI", "score": 96})]),
        Some(&["Updated Document 1".to_string()]),
    )
    .await?;
    let r = coll
        .get(
            Some(&["item1".to_string()]),
            None,
            None,
            None,
            None,
            None,
        )
        .await?;
    assert!(!r.ids.is_empty());
    coll.delete(
        Some(&["item1".to_string(), "item2".to_string()]),
        None,
        None,
    )
    .await?;
    assert_eq!(coll.count().await?, 0);

    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn database_operations() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
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
async fn sql_execution() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
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
async fn sql_error_handling() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    assert!(client.execute("SELECT * FROM non_existent_table").await.is_err());
    Ok(())
}
