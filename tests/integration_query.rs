//! Integration tests for collection query/get and filter behavior.

use anyhow::Result;
use seekdb_rs::{
    DistanceMetric, DocFilter, Filter, HnswConfig, IncludeField, SeekDbError, ServerClient,
};
use serde_json::json;

mod common;
use common::{ConstantEmbedding, DummyEmbedding, load_config_for_integration, ts_suffix};

#[tokio::test]
async fn collection_query_and_filters() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_query_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("q_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::L2,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    // Insert a few records
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

    coll.add(&ids, Some(&embs), Some(&metas), Some(&docs))
        .await?;

    // get with metadata filter
    let where_meta = Filter::Gt {
        field: "score".into(),
        value: json!(15),
    };
    let got = coll
        .get(None, Some(&where_meta), None, None, None, None)
        .await?;
    assert!(got.ids.len() >= 1);

    // get with document filter
    let where_doc = DocFilter::Contains("rust".into());
    let got_doc = coll
        .get(None, None, Some(&where_doc), None, None, None)
        .await?;
    assert!(got_doc.ids.len() >= 1);

    // query_embeddings default include: documents+metadatas, no embeddings
    let q = vec![vec![0.0, 0.0, 0.0]];
    let qr = coll.query_embeddings(&q, 2, None, None, None).await?;
    assert_eq!(qr.ids.len(), 1);
    assert_eq!(qr.distances.as_ref().unwrap()[0].len(), 2);
    assert!(qr.documents.as_ref().is_some());
    assert!(qr.metadatas.as_ref().is_some());
    assert!(qr.embeddings.is_none());

    // query_embeddings with embeddings included
    let qr2 = coll
        .query_embeddings(
            &q,
            2,
            None,
            None,
            Some(&[
                IncludeField::Documents,
                IncludeField::Metadatas,
                IncludeField::Embeddings,
            ]),
        )
        .await?;
    assert!(qr2.embeddings.as_ref().is_some());

    // README-style `Filter::In` metadata filter.
    let where_in = Filter::In {
        field: "tag".into(),
        values: vec![json!("x")],
    };
    let got_in = coll
        .get(None, Some(&where_in), None, None, None, None)
        .await?;
    assert!(got_in.ids.len() >= 1);

    // README-style `DocFilter::Regex` document filter.
    let where_doc_regex = DocFilter::Regex("rust".into());
    let got_regex = coll
        .get(None, None, Some(&where_doc_regex), None, None, None)
        .await?;
    assert!(got_regex.ids.len() >= 1);

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// query_texts should embed queries via embedding_function and reuse query_embeddings path.
#[tokio::test]
async fn collection_query_texts_with_embedding_function() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_qtexts_ok_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("qtexts_ok_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let ef = ConstantEmbedding { value: 0.2, dim: 3 };
    let coll = client
        .create_collection::<ConstantEmbedding>(&coll_name, Some(hnsw), Some(ef))
        .await?;

    let ids = vec!["qt1".to_string(), "qt2".to_string()];
    let docs = vec!["hello rust".to_string(), "hello seekdb".to_string()];
    // Use auto-embedding for adds (documents only).
    coll.add(&ids, None, None, Some(&docs)).await?;

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
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// query_texts should error when collection has no embedding_function.
#[tokio::test]
async fn collection_query_texts_not_implemented() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_qtexts_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

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
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}
