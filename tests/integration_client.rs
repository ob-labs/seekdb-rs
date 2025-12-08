//! Integration tests for client/admin APIs against a real SeekDB/OceanBase server.
//! These tests are skipped unless `SEEKDB_INTEGRATION=1` and SERVER_* env vars are set.

use std::sync::Arc;

use anyhow::Result;
use seekdb_rs::{AdminApi, AdminClient, ServerClient};

mod common;
use common::{load_config_for_integration, ts_suffix};

/// Smoke test for the README-style `ServerClient::connect` example.
#[tokio::test]
async fn client_connect_and_execute() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };

    let client = ServerClient::connect(
        &config.host,
        config.port,
        &config.tenant,
        &config.database,
        &config.user,
        &config.password,
    )
    .await?;

    // `SELECT 1` should succeed using the simple execute API.
    client.execute("SELECT 1").await?;
    Ok(())
}

/// Smoke test for the README-style `ServerClient::from_env` + `AdminClient` usage.
#[tokio::test]
async fn client_from_env_and_adminclient() -> Result<()> {
    let Some(_config) = load_config_for_integration() else {
        return Ok(());
    };

    // Uses the same SERVER_* env vars as `load_config_for_integration`.
    let client = ServerClient::from_env().await?;
    let admin = AdminClient::new(Arc::new(client));

    let db_name = format!("rs_readme_admin_{}", ts_suffix());

    admin.create_database(&db_name, None).await?;
    let db = admin.get_database(&db_name, None).await?;
    assert_eq!(db.name, db_name);

    let list = admin.list_databases(None, None, None).await?;
    assert!(list.iter().any(|d| d.name == db_name));

    admin.delete_database(&db_name, None).await?;
    Ok(())
}

/// Basic AdminClient database CRUD roundtrip.
#[tokio::test]
async fn admin_database_crud() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let client = ServerClient::from_config(config.clone()).await?;

    let db_name = format!("rs_admin_{}", ts_suffix());
    // Create
    client.create_database(&db_name, None).await?;
    // Get
    let db = client.get_database(&db_name, None).await?;
    assert_eq!(db.name, db_name);
    assert_eq!(db.tenant, Some(config.tenant.clone()));
    // List should contain it
    let list = client.list_databases(None, None, None).await?;
    assert!(list.iter().any(|d| d.name == db_name));
    // Delete
    client.delete_database(&db_name, None).await?;
    let list_after = client.list_databases(None, None, None).await?;
    assert!(!list_after.iter().any(|d| d.name == db_name));

    Ok(())
}

