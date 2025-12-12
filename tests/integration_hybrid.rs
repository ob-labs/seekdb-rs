//! Integration tests for hybrid_search and hybrid_search_advanced.

use anyhow::Result;
use seekdb_rs::{
    DistanceMetric, DocFilter, Embedding, Filter, HnswConfig, IncludeField, SeekDbError,
    ServerClient,
    collection::{HybridKnn, HybridQuery, HybridRank},
};
use serde_json::json;

mod common;
use common::{ConstantEmbedding, DummyEmbedding, load_config_for_integration, ts_suffix};

/// Hybrid search should succeed when using embedding_function for query text.
#[tokio::test]
async fn collection_hybrid_search_basic() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_hybrid_ok_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("hybrid_ok_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let ef = ConstantEmbedding { value: 0.3, dim: 3 };
    let coll = client
        .create_collection::<ConstantEmbedding>(&coll_name, Some(hnsw), Some(ef))
        .await?;

    // Insert a few docs via auto-embedding.
    let ids = vec!["hy1".to_string(), "hy2".to_string(), "hy3".to_string()];
    let docs = vec![
        "rust hybrid search".to_string(),
        "seekdb vector".to_string(),
        "other text".to_string(),
    ];
    coll.add(&ids, None, None, Some(&docs)).await?;

    let qr = coll
        .hybrid_search(
            &["rust".to_string()],
            None,
            None,
            None,
            3,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;

    assert_eq!(qr.ids.len(), 1);
    assert!(!qr.ids[0].is_empty(), "expected at least one hybrid result");

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// High-level hybrid_search with KNN-only configuration using precomputed query_embeddings.
#[tokio::test]
async fn collection_hybrid_search_advanced_vector_only() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_hybrid_adv_vec_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("hybrid_adv_vec_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    // Insert a few records with explicit embeddings so that vector search is meaningful.
    let ids = vec![
        format!("hv1_{}", ts_suffix()),
        format!("hv2_{}", ts_suffix()),
        format!("hv3_{}", ts_suffix()),
    ];
    let docs = vec![
        "vector item one".to_string(),
        "vector item two".to_string(),
        "vector item three".to_string(),
    ];
    let embs: Vec<Embedding> = vec![
        vec![1.0_f32, 2.0_f32, 3.0_f32],
        vec![1.1_f32, 2.1_f32, 3.1_f32],
        vec![5.0_f32, 5.0_f32, 5.0_f32],
    ];
    coll.add(&ids, Some(&embs), None, Some(&docs)).await?;

    // Use a query embedding close to the first two vectors.
    let query_vec: Embedding = vec![1.05_f32, 2.05_f32, 3.05_f32];
    let knn = HybridKnn {
        query_texts: None,
        query_embeddings: Some(vec![query_vec]),
        where_meta: None,
        n_results: Some(3),
    };

    let qr = coll
        .hybrid_search_advanced(
            None,
            Some(knn),
            None,
            3,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;

    assert_eq!(qr.ids.len(), 1);
    assert!(
        !qr.ids[0].is_empty(),
        "expected at least one result from advanced KNN-only hybrid_search"
    );

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// High-level hybrid_search combining full-text query, KNN, and RRF rank configuration.
#[tokio::test]
async fn collection_hybrid_search_advanced_query_knn_rank() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_hybrid_adv_full_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("hybrid_adv_full_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    // Insert a small corpus with metadata for filtering.
    let ids = vec![
        format!("hfv1_{}", ts_suffix()),
        format!("hfv2_{}", ts_suffix()),
        format!("hfv3_{}", ts_suffix()),
    ];
    let docs = vec![
        "machine learning with rust".to_string(),
        "python data science".to_string(),
        "machine learning basics".to_string(),
    ];
    let embs: Vec<Embedding> = vec![
        vec![1.0_f32, 2.0_f32, 3.0_f32],
        vec![0.0_f32, 0.0_f32, 1.0_f32],
        vec![1.1_f32, 2.1_f32, 3.1_f32],
    ];
    let metas = vec![
        json!({"category": "AI", "score": 95}),
        json!({"category": "Programming", "score": 80}),
        json!({"category": "AI", "score": 90}),
    ];
    coll.add(&ids, Some(&embs), Some(&metas), Some(&docs))
        .await?;

    // Build query: full-text "machine" with metadata filter category == "AI".
    let where_doc = DocFilter::Contains("machine".to_string());
    let where_meta = Filter::Eq {
        field: "category".to_string(),
        value: json!("AI"),
    };
    let query = HybridQuery {
        where_meta: Some(where_meta),
        where_doc: Some(where_doc),
    };

    // Build knn: query vector close to first/third embeddings, with score >= 90.
    let knn_where_meta = Filter::Gte {
        field: "score".to_string(),
        value: json!(90),
    };
    let knn = HybridKnn {
        query_texts: None,
        query_embeddings: Some(vec![vec![1.05_f32, 2.05_f32, 3.05_f32]]),
        where_meta: Some(knn_where_meta),
        n_results: Some(3),
    };

    let rank = HybridRank::Rrf {
        rank_window_size: Some(60),
        rank_constant: Some(60),
    };

    let qr = coll
        .hybrid_search_advanced(
            Some(query),
            Some(knn),
            Some(rank),
            3,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;

    assert_eq!(qr.ids.len(), 1);
    assert!(
        !qr.ids[0].is_empty(),
        "expected at least one result from advanced hybrid_search with query+knn+rank"
    );

    // All returned metadatas should satisfy category == "AI".
    if let Some(metas_out) = qr.metadatas.as_ref() {
        for meta in &metas_out[0] {
            if !meta.is_null() {
                assert_eq!(meta["category"], json!("AI"));
            }
        }
    }

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}

/// Verify that the hybrid_search API errors when missing embedding function for text queries.
#[tokio::test]
async fn collection_hybrid_search_not_implemented() -> Result<()> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let admin = ServerClient::from_config(config.clone()).await?;
    let db_name = format!("rs_hybrid_{}", ts_suffix());
    admin.create_database(&db_name, None).await?;

    let mut db_config = config.clone();
    db_config.database = db_name.clone();
    let client = ServerClient::from_config(db_config).await?;

    let coll_name = format!("hybrid_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;

    let res = coll
        .hybrid_search(&["query text".to_string()], None, None, None, 10, None)
        .await;

    assert!(matches!(res, Err(SeekDbError::Embedding(_))));

    client.delete_collection(&coll_name).await.ok();
    admin.delete_database(&db_name, None).await.ok();
    Ok(())
}
