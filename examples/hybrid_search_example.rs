//! Hybrid Search Example: query_texts() vs hybrid_search_advanced()
//!
//! Key advantages of hybrid_search_advanced():
//! - Combines full-text and vector search with independent filters
//! - RRF (Reciprocal Rank Fusion) for result ranking
//! - Better recall for keyword + semantic queries
//!
//! Run: `cargo run --example hybrid_search_example --no-default-features --features embedded,embedding`
//! Or with server: `cargo run --example hybrid_search_example` (set SEEKDB_* / host/port env or builder).

use anyhow::Result;
use seekdb_rs::{
    collection::{HybridKnn, HybridQuery, HybridRank},
    Client, DefaultEmbedding, DistanceMetric, DocFilter, EmbeddingFunction, Filter, HnswConfig,
    IncludeField,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
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

    let collection_name = "hybrid_search_example";
    let ef = DefaultEmbedding::new()?;
    let hnsw = HnswConfig {
        dimension: ef.dimension() as u32,
        distance: DistanceMetric::Cosine,
    };
    let collection = client
        .get_or_create_collection(collection_name, Some(hnsw), Some(ef))
        .await?;

    let documents = vec![
        "Machine learning is revolutionizing artificial intelligence and data science".to_string(),
        "Python programming language is essential for machine learning developers".to_string(),
        "Deep learning neural networks enable advanced AI applications".to_string(),
        "Data science combines statistics, programming, and domain expertise".to_string(),
        "Natural language processing uses machine learning to understand text".to_string(),
    ];
    let metadatas = [
        json!({"category": "AI", "topic": "machine learning", "year": 2023, "popularity": 95}),
        json!({"category": "Programming", "topic": "python", "year": 2023, "popularity": 88}),
        json!({"category": "AI", "topic": "deep learning", "year": 2024, "popularity": 92}),
        json!({"category": "Data Science", "topic": "data analysis", "year": 2023, "popularity": 85}),
        json!({"category": "AI", "topic": "nlp", "year": 2024, "popularity": 90}),
    ];
    let ids: Vec<String> = (0..documents.len()).map(|i| format!("doc_{}", i + 1)).collect();

    collection
        .add(&ids, None, Some(&metadatas), Some(&documents))
        .await?;

    println!("{}", "=".repeat(80));
    println!("SCENARIO 1: Keyword + Semantic Search");
    println!("{}", "=".repeat(80));
    println!("Goal: Find documents similar to 'AI research' AND containing 'machine learning'\n");

    // query_texts: single vector search with where_document filter
    let query_result = collection
        .query_texts(
            &["AI research".to_string()],
            5,
            None,
            Some(&DocFilter::Contains("machine learning".to_string())),
            Some(&[IncludeField::Documents]),
        )
        .await?;

    // hybrid_search_advanced: full-text branch + vector branch, then RRF
    let hybrid_query = HybridQuery {
        where_meta: None,
        where_doc: Some(DocFilter::Contains("machine learning".to_string())),
    };
    let hybrid_knn = HybridKnn {
        query_texts: Some(vec!["AI research".to_string()]),
        query_embeddings: None,
        where_meta: None,
        n_results: Some(10),
    };
    let hybrid_result = collection
        .hybrid_search_advanced(
            Some(hybrid_query),
            Some(hybrid_knn),
            Some(HybridRank::Rrf {
                rank_window_size: None,
                rank_constant: None,
            }),
            5,
            Some(&[IncludeField::Documents]),
        )
        .await?;

    println!("query_texts() result count: {}", query_result.ids[0].len());
    if let Some(docs) = &query_result.documents {
        for (i, id) in query_result.ids[0].iter().enumerate() {
            if let Some(d) = docs[0].get(i) {
                println!("  {}: {}...", id, d.chars().take(50).collect::<String>());
            }
        }
    }
    println!("\nhybrid_search_advanced() result count: {}", hybrid_result.ids[0].len());
    if let Some(docs) = &hybrid_result.documents {
        for (i, id) in hybrid_result.ids[0].iter().enumerate() {
            if let Some(d) = docs[0].get(i) {
                println!("  {}: {}...", id, d.chars().take(50).collect::<String>());
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("SCENARIO 2: Independent Filters (full-text vs vector)");
    println!("{}", "=".repeat(80));
    println!("Goal: Full-text='neural' (year=2024) + Vector='deep learning' (popularity>=90)\n");

    let query_filter = Filter::And(vec![
        Filter::Eq {
            field: "year".into(),
            value: json!(2024),
        },
        Filter::Gte {
            field: "popularity".into(),
            value: json!(90),
        },
    ]);
    let _query_result2 = collection
        .query_texts(
            &["deep learning".to_string()],
            5,
            Some(&query_filter),
            Some(&DocFilter::Contains("neural".to_string())),
            None,
        )
        .await?;

    let hybrid_query2 = HybridQuery {
        where_meta: Some(Filter::Eq {
            field: "year".into(),
            value: json!(2024),
        }),
        where_doc: Some(DocFilter::Contains("neural".to_string())),
    };
    let hybrid_knn2 = HybridKnn {
        query_texts: Some(vec!["deep learning".to_string()]),
        query_embeddings: None,
        where_meta: Some(Filter::Gte {
            field: "popularity".into(),
            value: json!(90),
        }),
        n_results: Some(10),
    };
    let _hybrid_result2 = collection
        .hybrid_search_advanced(
            Some(hybrid_query2),
            Some(hybrid_knn2),
            Some(HybridRank::Rrf {
                rank_window_size: None,
                rank_constant: None,
            }),
            5,
            None,
        )
        .await?;

    println!("With hybrid_search_advanced(), full-text and vector can use different filters.\n");

    client.delete_collection(collection_name).await?;
    println!("Deleted collection '{}'", collection_name);

    Ok(())
}
