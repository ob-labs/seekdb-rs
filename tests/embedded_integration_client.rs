//! Integration tests for embedded client/admin APIs.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::{AdminApi, Client, AdminClient, EmbeddedClient, SeekDbClient};

#[cfg(feature = "embedded")]
#[path = "embedded/common.rs"]
mod common;
#[cfg(feature = "embedded")]
use common::{load_config_for_integration, shared_db_dir, ts_suffix, ConstantEmbedding};

#[cfg(feature = "embedded")]
fn main() {
    common::run_embedded_tests(run_tests);
}

#[cfg(feature = "embedded")]
async fn run_tests() -> Result<()> {
    client_connect_and_execute().await?;
    client_builder_connect_and_execute().await?;
    client_from_env_and_execute().await?;
    client_trait_methods().await?;
    client_generic_trait().await?;
    client_and_admin_builders().await?;
    admin_database_crud().await?;
    collection_management().await?;
    Ok(())
}

/// Client::builder().path() + execute (mirrors server client_connect_and_execute).
#[cfg(feature = "embedded")]
async fn client_connect_and_execute() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    client.execute("SELECT 1", None).await?;
    Ok(())
}

/// EmbeddedClient::builder().db_dir() + execute (mirrors server client_builder_connect_and_execute).
#[cfg(feature = "embedded")]
async fn client_builder_connect_and_execute() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = EmbeddedClient::builder()
        .db_dir(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    client.execute("SELECT 1").await?;
    Ok(())
}

/// EmbeddedClient::builder().from_env() + execute (mirrors server client_from_env_and_admin_api).
#[cfg(feature = "embedded")]
async fn client_from_env_and_execute() -> Result<()> {
    let db_dir = shared_db_dir();
    unsafe {
        std::env::set_var("SEEKDB_EMBEDDED_INTEGRATION", "1");
        std::env::set_var("EMBEDDED_DB_DIR", db_dir.to_string_lossy().as_ref());
        std::env::set_var("EMBEDDED_DATABASE", "test");
    }
    let _config = load_config_for_integration().expect("env set above");
    let client = EmbeddedClient::builder()
        .from_env()
        .expect("env set above")
        .build()
        .await?;
    client.execute("SELECT 1").await?;
    assert!(!client.database().is_empty());
    Ok(())
}

/// Client + SeekDbClient: database(), list_collections(), count_collection().
#[cfg(feature = "embedded")]
async fn client_trait_methods() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    assert_eq!(client.database(), "test");
    let _ = client.list_collections().await?;
    let _ = client.count_collection().await?;
    Ok(())
}

/// Generic fn over SeekDbClient (execute, list_collections).
#[cfg(feature = "embedded")]
async fn client_generic_trait() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = EmbeddedClient::builder()
        .db_dir(db_dir.to_string_lossy().as_ref())
        .database("test")
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

/// Client + AdminClient builders, database CRUD via admin.
#[cfg(feature = "embedded")]
async fn client_and_admin_builders() -> Result<()> {
    let db_dir = shared_db_dir();
    let path = db_dir.to_string_lossy();
    let client = Client::builder()
        .path(path.as_ref())
        .database("test")
        .build()
        .await?;
    assert_eq!(client.database(), "test");
    client.execute("SELECT 1", None).await?;
    let admin = AdminClient::builder()
        .path(path.as_ref())
        .build()
        .await?;
    let db_name = format!("rs_unified_admin_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;
    let list = admin.list_databases(None, None, None).await?;
    assert!(list.iter().any(|d| d.name == db_name));
    admin.delete_database(&db_name, None).await?;
    Ok(())
}

/// Client: create_database, get_database, list_databases, delete_database.
#[cfg(feature = "embedded")]
async fn admin_database_crud() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let db_name = format!("rs_embedded_admin_{}", ts_suffix());
    client.create_database(&db_name, None).await?;
    let db = client.get_database(&db_name, None).await?;
    assert_eq!(db.name, db_name);
    let list = client.list_databases(None, None, None).await?;
    assert!(list.iter().any(|d| d.name == db_name));
    client.delete_database(&db_name, None).await?;
    let list_after = client.list_databases(None, None, None).await?;
    assert!(!list_after.iter().any(|d| d.name == db_name));
    Ok(())
}

/// Client: list_collections, has_collection, create/get/delete_collection (mirrors server collection_* in integration_collection_dml).
#[cfg(feature = "embedded")]
async fn collection_management() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("test_coll_{}", ts_suffix());
    let _ = client.list_collections().await?;
    let exists = client.has_collection(&coll_name).await?;
    assert!(!exists);
    let hnsw = seekdb_rs::HnswConfig {
        dimension: 3,
        distance: seekdb_rs::DistanceMetric::Cosine,
    };
    let embedding = ConstantEmbedding { value: 0.1, dim: 3 };
    let coll = client
        .create_collection(&coll_name, Some(hnsw), Some(embedding))
        .await?;
    assert_eq!(coll.name(), &coll_name);
    let names = client.list_collections().await?;
    assert!(names.contains(&coll_name));
    let coll2 = client.get_collection::<common::DummyEmbedding>(&coll_name, None).await?;
    assert_eq!(coll2.name(), &coll_name);
    client.delete_collection(&coll_name).await?;
    let exists_after = client.has_collection(&coll_name).await?;
    assert!(!exists_after);
    Ok(())
}
