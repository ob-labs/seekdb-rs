//! Integration tests for embedded hybrid_search operations.
#![cfg_attr(not(feature = "embedded"), allow(dead_code))]

#[cfg(not(feature = "embedded"))]
fn main() {}

#[cfg(feature = "embedded")]
use anyhow::Result;
#[cfg(feature = "embedded")]
use seekdb_rs::{
    collection::{HybridKnn, HybridQuery, HybridRank},
    AddBatch, Client, DocFilter, DistanceMetric, Embedding, Filter, HnswConfig, IncludeField,
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
    collection_hybrid_search_basic().await?;
    collection_hybrid_search_advanced_vector_only().await?;
    collection_hybrid_search_advanced_query_knn_rank().await?;
    collection_hybrid_search_not_implemented().await?;
    Ok(())
}

/// Hybrid search should succeed when using embedding_function for query text.
#[cfg(feature = "embedded")]
async fn collection_hybrid_search_basic() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("hybrid_ok_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let ef = ConstantEmbedding { value: 0.3, dim: 3 };
    let coll = client
        .create_collection::<ConstantEmbedding>(&coll_name, Some(hnsw), Some(ef))
        .await?;
    let ids = vec!["hy1".to_string(), "hy2".to_string(), "hy3".to_string()];
    let docs = vec![
        "rust hybrid search".to_string(),
        "seekdb vector".to_string(),
        "other text".to_string(),
    ];
    coll.add_batch(AddBatch::new(&ids).documents(&docs)).await?;
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
    Ok(())
}

/// High-level hybrid_search with KNN-only configuration using precomputed query_embeddings.
#[cfg(feature = "embedded")]
async fn collection_hybrid_search_advanced_vector_only() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("hybrid_adv_vec_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
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
    coll.add_batch(AddBatch::new(&ids).embeddings(&embs).documents(&docs))
        .await?;
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
    Ok(())
}

/// High-level hybrid_search combining full-text query, KNN, and RRF rank configuration.
#[cfg(feature = "embedded")]
async fn collection_hybrid_search_advanced_query_knn_rank() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
    let coll_name = format!("hybrid_adv_full_coll_{}", ts_suffix());
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    let coll = client
        .create_collection::<DummyEmbedding>(&coll_name, Some(hnsw), None::<DummyEmbedding>)
        .await?;
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
    coll.add_batch(
        AddBatch::new(&ids)
            .embeddings(&embs)
            .metadatas(&metas)
            .documents(&docs),
    )
    .await?;
    let where_doc = DocFilter::Contains("machine".to_string());
    let where_meta = Filter::Eq {
        field: "category".to_string(),
        value: json!("AI"),
    };
    let query = HybridQuery {
        where_meta: Some(where_meta),
        where_doc: Some(where_doc),
    };
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
    Ok(())
}

/// Verify that hybrid_search with text queries errors when collection has no embedding function.
#[cfg(feature = "embedded")]
async fn collection_hybrid_search_not_implemented() -> Result<()> {
    let db_dir = shared_db_dir();
    let client = Client::builder()
        .path(db_dir.to_string_lossy().as_ref())
        .database("test")
        .build()
        .await?;
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
    Ok(())
}
