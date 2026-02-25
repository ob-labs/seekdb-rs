//! Comprehensive Example: Complete guide to seekdb-rs features
//!
//! Demonstrates: client connection, collection management, DML (add/update/upsert/delete),
//! DQL (query, get, hybrid_search), filters, and collection info.
//!
//! Run: `cargo run --example complete_example --no-default-features --features embedded`
//! Or with server: `cargo run --example complete_example` (set SEEKDB_* / host/port env or builder).

use anyhow::Result;
use seekdb_rs::{
    collection::{HybridKnn, HybridRank},
    Client, DistanceMetric, DocFilter, Filter, HnswConfig, IncludeField,
};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

/// Minimal embedding that returns deterministic pseudo-random vectors (no network/model).
struct ExampleEmbedding {
    dim: usize,
    seed: AtomicU64,
}

impl ExampleEmbedding {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            seed: AtomicU64::new(42),
        }
    }
    fn next_vec(&self) -> Vec<f32> {
        let s = self.seed.fetch_add(1, Ordering::Relaxed);
        (0..self.dim)
            .map(|i| {
                let x = (s as u64).wrapping_add(i as u64).wrapping_mul(0x9e3779b97f4a7c15);
                ((x % 1000) as f32) / 1000.0
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl seekdb_rs::EmbeddingFunction for ExampleEmbedding {
    async fn embed_documents(
        &self,
        docs: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, seekdb_rs::SeekDbError> {
        Ok((0..docs.len()).map(|_| self.next_vec()).collect())
    }
    fn dimension(&self) -> usize {
        self.dim
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let dim = 8_usize;

    // ============================================================================
    // PART 1: CLIENT CONNECTION
    // ============================================================================
    // Embedded mode (local seekdb) – default for this example.
    let client = Client::builder()
        .path("./seekdb.db")
        .database("test")
        .build()
        .await?;

    // Server mode (remote SeekDB/OceanBase). Uncomment and comment out the block above to use.
    // tenant is optional (default "sys"); add .tenant("your_tenant") if needed.
    // let client = Client::builder()
    //     .host("127.0.0.1")
    //     .port(2881)
    //     .database("test")
    //     .user("root")
    //     .password("")
    //     .build()
    //     .await?;

    // ============================================================================
    // PART 2: COLLECTION MANAGEMENT
    // ============================================================================
    let collection_name = "complete_example";
    let ef = ExampleEmbedding::new(dim);
    let hnsw = HnswConfig {
        dimension: dim as u32,
        distance: DistanceMetric::Cosine,
    };
    let collection = client
        .get_or_create_collection(collection_name, Some(hnsw), Some(ef))
        .await?;

    let exists = client.has_collection(collection_name).await?;
    assert!(exists);
    let _retrieved = client
        .get_collection::<ExampleEmbedding>(collection_name, None)
        .await?;
    let _all = client.list_collections().await?;

    // ============================================================================
    // PART 3: DML - ADD
    // ============================================================================
    let ef2 = ExampleEmbedding::new(dim);
    let ids: Vec<String> = (0..5).map(|i| format!("doc_{}", i)).collect();
    let documents: Vec<String> = vec![
        "Machine learning is transforming the way we solve problems".into(),
        "Python programming language is widely used in data science".into(),
        "Vector databases enable efficient similarity search".into(),
        "Neural networks mimic the structure of the human brain".into(),
        "Natural language processing helps computers understand human language".into(),
    ];
    let embeddings: Vec<Vec<f32>> = (0..5).map(|_| ef2.next_vec()).collect();
    let metadatas = [
        json!({"category": "AI", "score": 95, "tag": "ml", "year": 2023}),
        json!({"category": "Programming", "score": 88, "tag": "python", "year": 2022}),
        json!({"category": "Database", "score": 92, "tag": "vector", "year": 2023}),
        json!({"category": "AI", "score": 90, "tag": "neural", "year": 2022}),
        json!({"category": "NLP", "score": 87, "tag": "language", "year": 2023}),
    ];

    collection
        .add(
            &ids,
            Some(&embeddings),
            Some(&metadatas),
            Some(&documents),
        )
        .await?;
    println!("Added {} items to collection", ids.len());

    // ============================================================================
    // PART 4: DML - UPDATE
    // ============================================================================
    collection
        .update(
            &[ids[0].clone()],
            None,
            Some(&[json!({"category": "AI", "score": 98, "tag": "ml", "year": 2024, "updated": true})]),
            None,
        )
        .await?;

    // ============================================================================
    // PART 5: DQL - QUERY (vector similarity)
    // ============================================================================
    let query_vec = embeddings[0].clone();
    let results = collection
        .query_embeddings(
            &[query_vec],
            3,
            None,
            None,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;
    println!("Query (vector): {} result(s)", results.ids[0].len());

    // Query with metadata filter
    let where_meta = Filter::Gte {
        field: "score".into(),
        value: json!(90),
    };
    let results2 = collection
        .query_embeddings(
            &[embeddings[0].clone()],
            5,
            Some(&where_meta),
            None,
            None,
        )
        .await?;
    println!("Query (score >= 90): {} result(s)", results2.ids[0].len());

    // Query with $in
    let where_in = Filter::In {
        field: "tag".into(),
        values: vec![json!("ml"), json!("python"), json!("neural")],
    };
    let _results3 = collection
        .query_embeddings(
            &[embeddings[0].clone()],
            5,
            Some(&where_in),
            None,
            None,
        )
        .await?;

    // Query with document filter
    let where_doc = DocFilter::Contains("machine learning".into());
    let _results4 = collection
        .query_embeddings(
            &[embeddings[0].clone()],
            5,
            None,
            Some(&where_doc),
            None,
        )
        .await?;

    // ============================================================================
    // PART 6: DQL - GET
    // ============================================================================
    let get_result = collection
        .get(
            Some(&[ids[0].clone(), ids[1].clone()]),
            None,
            None,
            None,
            None,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;
    println!("Get by ids: {} item(s)", get_result.ids.len());

    let get_where = collection
        .get(
            None,
            Some(&Filter::Eq {
                field: "category".into(),
                value: json!("AI"),
            }),
            None,
            Some(5),
            Some(0),
            None,
        )
        .await?;
    println!("Get by where (category=AI): {} item(s)", get_where.ids.len());

    // ============================================================================
    // PART 7: HYBRID SEARCH (KNN + optional full-text)
    // ============================================================================
    let knn = HybridKnn {
        query_embeddings: Some(vec![embeddings[0].clone()]),
        query_texts: None,
        where_meta: Some(Filter::Gte {
            field: "year".into(),
            value: json!(2022),
        }),
        n_results: Some(10),
    };
    let hybrid_results = collection
        .hybrid_search_advanced(
            None,
            Some(knn),
            Some(HybridRank::Rrf {
                rank_window_size: None,
                rank_constant: None,
            }),
            5,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;
    println!("Hybrid search (KNN + filter): {} result(s)", hybrid_results.ids[0].len());

    // ============================================================================
    // PART 8: COLLECTION INFO & CLEANUP
    // ============================================================================
    let count = collection.count().await?;
    println!("Collection count: {}", count);

    let peek_result = collection.peek(5).await?;
    println!("Peek: {} item(s)", peek_result.ids.len());

    let coll_count = client.count_collection().await?;
    println!("Database collection count: {}", coll_count);

    // Delete by ids
    collection
        .delete(
            Some(&[ids[3].clone(), ids[4].clone()]),
            None,
            None,
        )
        .await?;

    client.delete_collection(collection_name).await?;
    println!("Deleted collection '{}'", collection_name);

    Ok(())
}
