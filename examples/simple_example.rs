//! Simple Example: Basic usage of seekdb-rs with embedding functions
//!
//! Demonstrates:
//! 1. Create a client connection (embedded mode)
//! 2. Create a collection with default embedding function
//! 3. Add data using documents (embeddings auto-generated)
//! 4. Query using query texts (embeddings auto-generated)
//! 5. Print query results
//!
//! Run: `cargo run --example simple_example --no-default-features --features embedded,embedding`
//! Or with server: `cargo run --example simple_example` (set SEEKDB_* / host/port env or builder).

use anyhow::Result;
use seekdb_rs::EmbeddingFunction;
use seekdb_rs::{
    Client, DefaultEmbedding, DistanceMetric, HnswConfig, IncludeField,
};

#[tokio::main]
async fn main() -> Result<()> {
    // ==================== Step 1: Create Client Connection ====================
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

    // ==================== Step 2: Create a Collection with Embedding Function ====================
    let collection_name = "simple_example";
    let ef = DefaultEmbedding::new()?;
    let hnsw = HnswConfig {
        dimension: ef.dimension() as u32,
        distance: DistanceMetric::Cosine,
    };
    let collection = client
        .create_collection(collection_name, Some(hnsw), Some(ef))
        .await?;

    println!(
        "Created collection '{}' with dimension: {}",
        collection_name,
        collection.dimension()
    );

    // ==================== Step 3: Add Data to Collection ====================
    let documents: Vec<String> = vec![
        "Machine learning is a subset of artificial intelligence".into(),
        "Python is a popular programming language".into(),
        "Vector databases enable semantic search".into(),
        "Neural networks are inspired by the human brain".into(),
        "Natural language processing helps computers understand text".into(),
    ];
    let ids: Vec<String> = ["id1", "id2", "id3", "id4", "id5"]
        .into_iter()
        .map(String::from)
        .collect();
    let metadatas = [
        serde_json::json!({"category": "AI", "index": 0}),
        serde_json::json!({"category": "Programming", "index": 1}),
        serde_json::json!({"category": "Database", "index": 2}),
        serde_json::json!({"category": "AI", "index": 3}),
        serde_json::json!({"category": "NLP", "index": 4}),
    ];

    collection
        .add(&ids, None, Some(&metadatas), Some(&documents))
        .await?;

    println!("\nAdded {} documents to collection (embeddings auto-generated)", documents.len());

    // ==================== Step 4: Query the Collection ====================
    let query_text = "artificial intelligence and machine learning";
    let results = collection
        .query_texts(
            &[query_text.to_string()],
            3,
            None,
            None,
            Some(&[IncludeField::Documents, IncludeField::Metadatas]),
        )
        .await?;

    println!("\nQuery: '{}'", query_text);
    println!("Query results: {} items found", results.ids[0].len());

    for (i, id) in results.ids[0].iter().enumerate() {
        println!("\nResult {}:", i + 1);
        println!("  ID: {}", id);
        if let Some(dists) = &results.distances {
            if let Some(d) = dists[0].get(i) {
                println!("  Distance: {:.4}", d);
            }
        }
        if let Some(docs) = &results.documents {
            if let Some(d) = docs[0].get(i) {
                println!("  Document: {}", d);
            }
        }
        if let Some(metas) = &results.metadatas {
            if let Some(m) = metas[0].get(i) {
                println!("  Metadata: {}", m);
            }
        }
    }

    // ==================== Step 5: Cleanup ====================
    client.delete_collection(collection_name).await?;
    println!("\nDeleted collection '{}'", collection_name);

    Ok(())
}
