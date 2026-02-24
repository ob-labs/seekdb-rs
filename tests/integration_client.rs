//! Integration tests for client/admin APIs against a real SeekDB/OceanBase server.
//! Use unified `Client` as entry (Client::builder().host() / from_server_config()).
//! These tests are skipped unless `SEEKDB_INTEGRATION=1` and SERVER_* env vars are set.

use anyhow::Result;
use seekdb_rs::{AdminApi, Client, ServerConfig};

mod common;
use common::{client_from_config, load_config_for_integration, ts_suffix};

/// Smoke test for unified Client (host => server).
#[tokio::test]
async fn client_connect_and_execute() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };

    let client = Client::builder()
        .host(&config.host)
        .port(config.port)
        .tenant(&config.tenant)
        .database(&config.database)
        .user(&config.user)
        .password(&config.password)
        .max_connections(config.max_connections)
        .build()
        .await?;

    client.execute("SELECT 1").await?;
    Ok(())
}

/// Smoke test for Client from env (server) and AdminApi on Client.
#[tokio::test]
async fn client_from_env_and_admin_api() -> Result<()> {
    let Some(_) = load_config_for_integration() else {
        return Ok(());
    };

    let config = ServerConfig::from_env()?;
    let client = client_from_config(config).await?;

    let db_name = format!("rs_readme_admin_{}", ts_suffix());

    client.create_database(&db_name, None).await?;
    let db = client.get_database(&db_name, None).await?;
    assert_eq!(db.name, db_name);

    let list = client.list_databases(None, None, None).await?;
    assert!(list.iter().any(|d| d.name == db_name));

    client.delete_database(&db_name, None).await?;
    Ok(())
}

/// Smoke test for Client::builder() fluent API (server).
#[tokio::test]
async fn client_builder_connect_and_execute() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };

    let client = Client::builder()
        .host(&config.host)
        .port(config.port)
        .tenant(&config.tenant)
        .database(&config.database)
        .user(&config.user)
        .password(&config.password)
        .max_connections(config.max_connections)
        .build()
        .await?;

    client.execute("SELECT 1").await?;
    Ok(())
}

/// Basic database CRUD roundtrip via unified Client (AdminApi).
#[tokio::test]
async fn admin_database_crud() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let client = client_from_config(config.clone()).await?;

    let db_name = format!("rs_admin_{}", ts_suffix());
    client.create_database(&db_name, None).await?;
    let db = client.get_database(&db_name, None).await?;
    assert_eq!(db.name, db_name);
    assert_eq!(db.tenant, Some(config.tenant.clone()));

    let list = client.list_databases(None, None, None).await?;
    assert!(list.iter().any(|d| d.name == db_name));

    client.delete_database(&db_name, None).await?;
    let list_after = client.list_databases(None, None, None).await?;
    assert!(!list_after.iter().any(|d| d.name == db_name));

    Ok(())
}
