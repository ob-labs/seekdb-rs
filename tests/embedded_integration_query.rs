//! Integration tests for embedded collection query/get operations.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::{
    AddBatch, Client, DocFilter, DistanceMetric, Filter, GetQuery, HnswConfig, IncludeField,
    SeekDbError,
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
    query_execute_and_fetch().await?;
    query_filter_where().await?;
    collection_query_and_filters().await?;
    collection_query_texts_with_embedding_function().await?;
    collection_query_texts_not_implemented().await?;
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

#[cfg(feature = "embedded")]
async fn collection_query_and_filters() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("q_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::L2,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
    let ids = vec!["qa1".to_string(), "qa2".to_string(), "qa3".to_string()];
    let embs = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    let docs = vec![
        "rust integration test".to_string(),
        "other document".to_string(),
        "rust and databases".to_string(),
    ];
    let metas = vec![
        json!({"score": 10, "tag": "x"}),
        json!({"score": 20, "tag": "y"}),
        json!({"score": 30, "tag": "x"}),
    ];
    coll.add_batch(
        AddBatch::new(&ids)
            .embeddings(&embs)
            .metadatas(&metas)
            .documents(&docs),
    )
    .await?;
    let where_meta = Filter::Gt {
        field: "score".into(),
        value: json!(15),
    };
    let got = coll
        .get_query(GetQuery::new().with_where_meta(&where_meta))
        .await?;
    assert!(got.ids.len() >= 1);
    let where_doc = DocFilter::Contains("rust".into());
    let got_doc = coll
        .get_query(GetQuery::new().with_where_doc(&where_doc))
        .await?;
    assert!(got_doc.ids.len() >= 1);
    let q = vec![vec![0.0, 0.0, 0.0]];
    let qr = coll.query_embeddings(&q, 2, None, None, None).await?;
    assert_eq!(qr.ids.len(), 1);
    assert!(qr.documents.as_ref().is_some());
    assert!(qr.metadatas.as_ref().is_some());
    let where_in = Filter::In {
        field: "tag".into(),
        values: vec![json!("x")],
    };
    let got_in = coll
        .get_query(GetQuery::new().with_where_meta(&where_in))
        .await?;
    assert!(got_in.ids.len() >= 1);
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_query_texts_with_embedding_function() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("qtexts_ok_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let ef = ConstantEmbedding { value: 0.2, dim: 3 };
    let coll = client
        .create_collection(&coll_name, Some(hnsw), Some(ef))
        .await?;
    let ids = vec!["qt1".to_string(), "qt2".to_string()];
    let docs = vec!["hello rust".to_string(), "hello seekdb".to_string()];
    coll.add_batch(AddBatch::new(&ids).documents(&docs)).await?;
    let qr = coll
        .query_texts(
            &["hello rust".to_string()],
            2,
            None,
            None,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;
    assert_eq!(qr.ids.len(), 1);
    assert!(!qr.ids[0].is_empty());
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}

#[cfg(feature = "embedded")]
async fn collection_query_texts_not_implemented() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("qtexts_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
    let res = coll
        .query_texts(
            &["some query".to_string()],
            5,
            None,
            None,
            Some(&[IncludeField::Documents]),
        )
        .await;
    assert!(matches!(res, Err(SeekDbError::Embedding(_))));
    client.delete_collection(&coll_name).await.ok();
    Ok(())
}
