#![cfg(feature = "sync")]
//! Integration tests for the synchronous (`sync` feature) wrapper APIs.
//! These tests are skipped unless `SEEKDB_INTEGRATION=1` and SERVER_* env vars are set.

use anyhow::Result;
use seekdb_rs::{DistanceMetric, HnswConfig, SyncServerClient};

mod common;
use common::{load_config_for_integration, ts_suffix, DummyEmbedding};

/// Basic DML roundtrip using the synchronous client and collection APIs.
#[test]
fn sync_collection_dml_roundtrip() -> Result<()> {
    let Some(mut config) = load_config_for_integration() else {
        return Ok(());
    };

    // Admin client on the default database.
    let admin = SyncServerClient::from_config(config.clone())?;
    let db_name = format!("rs_sync_dml_{}", ts_suffix());
    admin.create_database(&db_name, None)?;

    // Use the dedicated database for this test.
    config.database = db_name.clone();
    let client = SyncServerClient::from_config(config.clone())?;

    let coll_name = format!("sync_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };

    let coll = client.create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)?;

    let id1 = format!("sid1_{}", ts_suffix());
    let id2 = format!("sid2_{}", ts_suffix());

    coll.add(
        &[id1.clone(), id2.clone()],
        Some(&[vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]),
        None,
        Some(&["sdoc1".into(), "sdoc2".into()]),
    )?;

    let cnt = coll.count()?;
    assert_eq!(cnt, 2);

    // Cleanup
    client.delete_collection(&coll_name).ok();
    admin.delete_database(&db_name, None).ok();

    Ok(())
}

