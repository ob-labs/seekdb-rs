//! Integration tests for collection DML and metadata/upsert semantics.

use anyhow::Result;
use seekdb_rs::{DistanceMetric, Filter, HnswConfig, IncludeField, SeekDbError, ServerClient};
use serde_json::json;

mod common;
use common::{ConstantEmbedding, DummyEmbedding, load_config_for_integration, ts_suffix};

/// Creating a collection without HnswConfig should return a config error.
#[tokio::test]
async fn collection_create_without_hnsw_config_errors() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };

    let client = ServerClient::from_config(config).await?;
    let name = format!("no_cfg_coll_{}", ts_suffix());

    let res = client
        .create_collection::<DummyEmbedding>(&name, None, None::<DummyEmbedding>)
        .await;

    match res {
        Err(SeekDbError::Config(msg)) => {
            assert!(
                msg.contains("HnswConfig must be provided"),
                "unexpected config error message: {msg}"
            );
        }
        Ok(_) => panic!("expected SeekDbError::Config, got Ok(_)"),
        Err(e) => panic!("expected SeekDbError::Config, got different error: {e:?}"),
    }

    Ok(())
}

/// Invalid embedding dimension should surface as SeekDbError::InvalidInput.
#[tokio::test]
async fn collection_add_invalid_embedding_dimension_errors() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_invalid_dim_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("invalid_dim_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    let ids = vec!["id_invalid_dim".to_string()];
    // Deliberately wrong dimension (2 instead of 3)
    let bad_embs = vec![vec![1.0_f32, 2.0_f32]];

    let res = coll.add(&ids, Some(&bad_embs), None, None).await;
    match res {
        Err(SeekDbError::InvalidInput(msg)) => {
            assert!(
                msg.contains("embedding dimension"),
                "unexpected invalid-input message: {msg}"
            );
        }
        other => panic!("expected SeekDbError::InvalidInput, got: {:?}", other),
    }

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// Adding with documents only should auto-generate embeddings when embedding_function is present.
#[tokio::test]
async fn collection_add_with_auto_embedding() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_auto_emb_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("auto_emb_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let ef = ConstantEmbedding { value: 0.5, dim: 3 };
    let coll = client
        .create_collection::<ConstantEmbedding>(&coll_name, Some(hnsw), Some(ef))
        .await?;

    let ids = vec!["auto1".to_string(), "auto2".to_string()];
    let docs = vec!["hello rust".to_string(), "seekdb vector".to_string()];
    coll.add(&ids, None, None, Some(&docs)).await?;

    let got = coll
        .get(
            None,
            None,
            None,
            None,
            None,
            Some(&[
                IncludeField::Documents,
                IncludeField::Metadatas,
                IncludeField::Embeddings,
            ]),
        )
        .await?;

    assert_eq!(got.ids.len(), 2);
    assert_eq!(got.documents.as_ref().unwrap().len(), 2);
    let embs = got.embeddings.as_ref().unwrap();
    assert_eq!(embs.len(), 2);
    assert!(embs.iter().all(|e| e.len() == 3));
    assert!(
        embs.iter()
            .all(|e| e.iter().all(|v| (*v - 0.5).abs() < 1e-5))
    );

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// Mismatched ids/embeddings lengths should be rejected.
#[tokio::test]
async fn collection_add_length_mismatch_errors() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_len_mismatch_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("len_mismatch_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    let ids = vec!["id1".to_string(), "id2".to_string()];
    // Only one embedding for two ids.
    let embs = vec![vec![1.0_f32, 2.0_f32, 3.0_f32]];

    let res = coll.add(&ids, Some(&embs), None, None).await;
    match res {
        Err(SeekDbError::InvalidInput(msg)) => {
            assert!(
                msg.contains("embeddings length") && msg.contains("ids length"),
                "unexpected invalid-input message: {msg}"
            );
        }
        other => panic!("expected SeekDbError::InvalidInput, got: {:?}", other),
    }

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// Full DML roundtrip: add/update/upsert/delete/count/peek.
#[tokio::test]
async fn collection_dml_roundtrip() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    // Create a dedicated database for the test
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_dml_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    // Use the new database
    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    let id1 = format!("id1_{}", ts_suffix());
    let id2 = format!("id2_{}", ts_suffix());
    let id3 = format!("id3_{}", ts_suffix());

    // Add two items
    coll.add(
        &[id1.clone(), id2.clone()],
        Some(&[vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]),
        Some(&[json!({"category":"a"}), json!({"category":"b"})]),
        Some(&["doc1".into(), "doc2".into()]),
    )
    .await?;

    // Basic get
    let got = coll
        .get(Some(&[id1.clone()]), None, None, None, None, None)
        .await?;
    assert_eq!(got.ids.len(), 1);

    // Update metadata only
    coll.update(
        &[id1.clone()],
        None,
        Some(&[json!({"category":"a","updated":true})]),
        None,
    )
    .await?;

    // Upsert existing and new
    coll.upsert(
        &[id1.clone(), id3.clone()],
        Some(&[vec![1.0, 2.0, 3.0], vec![3.0, 3.0, 3.0]]),
        Some(&[json!({"category":"a2"}), json!({"category":"remove"})]),
        Some(&["doc1-up".into(), "doc3".into()]),
    )
    .await?;

    // Delete by id
    coll.delete(Some(&[id2.clone()]), None, None).await?;
    // Delete by metadata filter
    coll.delete(
        None,
        Some(&Filter::Eq {
            field: "category".into(),
            value: json!("remove"),
        }),
        None,
    )
    .await?;

    // Count and peek
    let cnt = coll.count().await?;
    assert!(cnt >= 1);
    let _ = coll.peek(5).await?;

    // Cleanup
    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// End-to-end flow mirroring the README quickstart.
#[tokio::test]
async fn collection_quickstart_like_flow() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };

    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_quickstart_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("quickstart_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };

    // Use the boxed trait-object type parameter as shown in README.
    let coll = client
        .create_collection::<Box<dyn seekdb_rs::EmbeddingFunction>>(
            &coll_name,
            Some(hnsw),
            None::<Box<dyn seekdb_rs::EmbeddingFunction>>,
        )
        .await?;

    let ids = vec!["id1".to_string(), "id2".to_string()];
    let embs = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
    let docs = vec!["doc1".to_string(), "doc2".to_string()];
    let metas = vec![json!({"score": 10}), json!({"score": 20})];

    coll.add(&ids, Some(&embs), Some(&metas), Some(&docs))
        .await?;

    let query = vec![vec![1.0, 2.0, 3.0]];
    let qr = coll
        .query_embeddings(
            &query,
            2,
            None,
            None,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;

    assert_eq!(qr.ids.len(), 1);
    assert_eq!(qr.ids[0].len(), 2);

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// Upsert metadata/doc/embeddings independently, keeping other fields intact.
#[tokio::test]
async fn collection_upsert_metadata_and_partial_fields() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_upsert_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("upsert_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    let id = format!("u1_{}", ts_suffix());

    // Seed record
    coll.add(
        &[id.clone()],
        Some(&[vec![1.0, 2.0, 3.0]]),
        Some(&[json!({"field": "orig", "cnt": 1})]),
        Some(&["orig_doc".to_string()]),
    )
    .await?;

    // 1) metadata-only upsert: update cnt, keep doc and embedding
    coll.upsert(
        &[id.clone()],
        None,
        Some(&[json!({"field": "orig", "cnt": 2})]),
        None,
    )
    .await?;

    let got1 = coll
        .get(Some(&[id.clone()]), None, None, None, None, None)
        .await?;
    assert_eq!(got1.documents.as_ref().unwrap()[0], "orig_doc");
    assert_eq!(got1.metadatas.as_ref().unwrap()[0]["cnt"], 2);

    // 2) document-only upsert: change doc, keep metadata and embedding
    coll.upsert(&[id.clone()], None, None, Some(&["new_doc".to_string()]))
        .await?;
    let got2 = coll
        .get(Some(&[id.clone()]), None, None, None, None, None)
        .await?;
    assert_eq!(got2.documents.as_ref().unwrap()[0], "new_doc");
    assert_eq!(got2.metadatas.as_ref().unwrap()[0]["cnt"], 2);

    // 3) embeddings-only upsert: change vector, keep doc and metadata
    coll.upsert(&[id.clone()], Some(&[vec![3.0, 2.0, 1.0]]), None, None)
        .await?;

    let got3 = coll
        .get(
            Some(&[id.clone()]),
            None,
            None,
            None,
            None,
            Some(&[
                IncludeField::Embeddings,
                IncludeField::Documents,
                IncludeField::Metadatas,
            ]),
        )
        .await?;
    assert_eq!(got3.documents.as_ref().unwrap()[0], "new_doc");
    assert_eq!(got3.metadatas.as_ref().unwrap()[0]["cnt"], 2);
    let emb = &got3.embeddings.as_ref().unwrap()[0];
    assert_eq!(emb.len(), 3);

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// Verify that deleting without any condition is rejected.
#[tokio::test]
async fn collection_delete_without_any_condition_errors() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_delete_guard_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("delete_guard_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    // Insert one record so that a blanket DELETE would be dangerous.
    coll.add(
        &[format!("dg_{}", ts_suffix())],
        Some(&[vec![1.0_f32, 2.0_f32, 3.0_f32]]),
        None,
        None,
    )
    .await?;

    let res = coll.delete(None, None, None).await;
    match res {
        Err(SeekDbError::InvalidInput(msg)) => {
            assert!(
                msg.contains("ids/where_meta/where_doc"),
                "unexpected invalid-input message: {msg}"
            );
        }
        other => panic!("expected SeekDbError::InvalidInput, got: {:?}", other),
    }

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// List collections and verify has_collection/get_collection metadata.
#[tokio::test]
async fn collection_list_and_has() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_list_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll1 = format!("list_coll1_{}", ts_suffix());
    let coll2 = format!("list_coll2_{}", ts_suffix());
    client
        .create_collection::<DummyEmbedding>(&coll1, Some(hnsw.clone()), None::<DummyEmbedding>)
        .await?;
    client
        .create_collection::<DummyEmbedding>(&coll2, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    let names = client.list_collections().await?;
    assert!(names.contains(&coll1));
    assert!(names.contains(&coll2));
    assert!(client.has_collection(&coll1).await?);
    assert!(!client.has_collection("no_such_collection").await?);

    // get_collection should pick up dimension + distance
    let coll = client
        .get_collection::<DummyEmbedding>(&coll1, None::<DummyEmbedding>)
        .await?;
    assert_eq!(coll.dimension(), 3);

    client.delete_collection(&coll1).await.ok();
    client.delete_collection(&coll2).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}
