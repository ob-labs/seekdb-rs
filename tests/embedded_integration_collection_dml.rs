//! Integration tests for embedded collection DML operations.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::{
    AddBatch, AdminApi, Client, DeleteQuery, DistanceMetric, Embedding, Filter, GetQuery,
    HnswConfig, IncludeField, SeekDbError, UpdateBatch, UpsertBatch,
};
#[cfg(feature = "embedded")]
use serde_json::json;
#[cfg(feature = "embedded")]
#[path = "embedded/common.rs"]
mod common;
#[cfg(feature = "embedded")]
use common::{shared_db_dir, ConstantEmbedding, DummyEmbedding, ts_suffix};

#[cfg(feature = "embedded")]
fn main() {
    common::run_embedded_tests(run_tests);
}

#[cfg(feature = "embedded")]
async fn run_tests() -> Result<()> {
    collection_create_without_hnsw_config_errors().await?;
    collection_add_invalid_embedding_dimension_errors().await?;
    collection_add_with_auto_embedding().await?;
    collection_add_length_mismatch_errors().await?;
    collection_dml_roundtrip().await?;
    collection_quickstart_like_flow().await?;
    collection_upsert_metadata_and_partial_fields().await?;
    collection_delete_without_any_condition_errors().await?;
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
async fn collection_add_invalid_embedding_dimension_errors() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("invalid_dim_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
    let ids = vec!["id_invalid_dim".to_string()];
    let bad_embs = vec![vec![1.0_f32, 2.0_f32]];
    let res = coll.add_batch(AddBatch::new(&ids).embeddings(&bad_embs)).await;
    match res {
        Err(SeekDbError::InvalidInput(msg)) => {
            assert!(msg.contains("embedding dimension"), "unexpected: {msg}");
        }
        other => panic!("expected SeekDbError::InvalidInput, got: {:?}", other),
    }
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_add_with_auto_embedding() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("auto_emb_coll_{}", ts_suffix());
    let ef = ConstantEmbedding { value: 0.5, dim: 3 };
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .get_or_create_collection(&coll_name, Some(hnsw), Some(ef))
        .await?;
    let ids = vec!["a1".to_string(), "a2".to_string()];
    let docs = vec!["doc a".to_string(), "doc b".to_string()];
    coll.add(&ids, None, None, Some(&docs)).await?;
    let n = coll.count().await?;
    assert_eq!(n, 2);
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_add_length_mismatch_errors() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("len_mismatch_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
    let ids = vec!["id1".to_string(), "id2".to_string()];
    let embs = vec![vec![1.0_f32, 2.0_f32, 3.0_f32]];
    let res = coll.add_batch(AddBatch::new(&ids).embeddings(&embs)).await;
    match res {
        Err(SeekDbError::InvalidInput(msg)) => {
            assert!(
                msg.contains("embeddings length") && msg.contains("ids length"),
                "unexpected: {msg}"
            );
        }
        other => panic!("expected SeekDbError::InvalidInput, got: {:?}", other),
    }
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_dml_roundtrip() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
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
    coll.add_batch(
        AddBatch::new(&[id1.clone(), id2.clone()])
            .embeddings(&[vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]])
            .metadatas(&[json!({"category":"a"}), json!({"category":"b"})])
            .documents(&["doc1".into(), "doc2".into()]),
    )
    .await?;
    let got = coll.get_query(GetQuery::by_ids(&[id1.clone()])).await?;
    assert_eq!(got.ids.len(), 1);
    coll.update_batch(
        UpdateBatch::new(&[id1.clone()]).metadatas(&[json!({"category":"a","updated":true})]),
    )
    .await?;
    coll.upsert_batch(
        UpsertBatch::new(&[id1.clone(), id3.clone()])
            .embeddings(&[vec![1.0, 2.0, 3.0], vec![3.0, 3.0, 3.0]])
            .metadatas(&[json!({"category":"a2"}), json!({"category":"remove"})])
            .documents(&["doc1-up".into(), "doc3".into()]),
    )
    .await?;
    coll.delete_query(DeleteQuery::by_ids(&[id2.clone()])).await?;
    coll.delete_query(DeleteQuery::new().with_where_meta(&Filter::Eq {
        field: "category".into(),
        value: json!("remove"),
    }))
    .await?;
    let cnt = coll.count().await?;
    assert!(cnt >= 1);
    let _ = coll.peek(5).await?;
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_quickstart_like_flow() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("quickstart_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
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
    coll.add_batch(
        AddBatch::new(&ids)
            .embeddings(&embs)
            .metadatas(&metas)
            .documents(&docs),
    )
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
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_upsert_metadata_and_partial_fields() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("upsert_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
    let id = format!("u1_{}", ts_suffix());
    coll.add_batch(
        AddBatch::new(&[id.clone()])
            .embeddings(&[vec![1.0, 2.0, 3.0]])
            .metadatas(&[json!({"field": "orig", "cnt": 1})])
            .documents(&["orig_doc".to_string()]),
    )
    .await?;
    coll.upsert_batch(
        UpsertBatch::new(&[id.clone()]).metadatas(&[json!({"field": "orig", "cnt": 2})]),
    )
    .await?;
    let got1 = coll.get_query(GetQuery::by_ids(&[id.clone()])).await?;
    assert_eq!(got1.documents.as_ref().unwrap()[0], "orig_doc");
    assert_eq!(got1.metadatas.as_ref().unwrap()[0]["cnt"], 2);
    coll.upsert_batch(
        UpsertBatch::new(&[id.clone()]).documents(&["new_doc".to_string()]),
    )
    .await?;
    let got2 = coll.get_query(GetQuery::by_ids(&[id.clone()])).await?;
    assert_eq!(got2.documents.as_ref().unwrap()[0], "new_doc");
    assert_eq!(got2.metadatas.as_ref().unwrap()[0]["cnt"], 2);
    coll.upsert_batch(UpsertBatch::new(&[id.clone()]).embeddings(&[vec![3.0, 2.0, 1.0]]))
        .await?;
    let got3 = coll
        .get_query(
            GetQuery::by_ids(&[id.clone()]).with_include(&[
                IncludeField::Embeddings,
                IncludeField::Documents,
                IncludeField::Metadatas,
            ]),
        )
        .await?;
    assert_eq!(got3.documents.as_ref().unwrap()[0], "new_doc");
    assert_eq!(got3.metadatas.as_ref().unwrap()[0]["cnt"], 2);
    assert_eq!(got3.embeddings.as_ref().unwrap()[0].len(), 3);
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_delete_without_any_condition_errors() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("delete_guard_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
    coll.add_batch(
        AddBatch::new(&[format!("dg_{}", ts_suffix())])
            .embeddings(&[vec![1.0_f32, 2.0_f32, 3.0_f32]]),
    )
    .await?;
    let res = coll.delete_query(DeleteQuery::new()).await;
    match res {
        Err(SeekDbError::InvalidInput(msg)) => {
            assert!(
                msg.contains("ids/where_meta/where_doc"),
                "unexpected: {msg}"
            );
        }
        other => panic!("expected SeekDbError::InvalidInput, got: {:?}", other),
    }
    client.delete_collection(&coll_name).await.ok();
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
    client.execute("CREATE TABLE IF NOT EXISTS test_table (id INT, name VARCHAR(100))", None).await?;
    client.execute("INSERT INTO test_table (id, name) VALUES (1, 'test1')", None).await?;
    let rows = client.fetch_all("SELECT id, name FROM test_table ORDER BY id", None).await?;
    assert_eq!(rows.len(), 1);
    client.execute("DROP TABLE IF EXISTS test_table", None).await?;
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
    assert!(client.execute("SELECT * FROM non_existent_table", None).await.is_err());
    Ok(())
}
