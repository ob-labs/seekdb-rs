use seekdb_rs::{
    AdminApi, AdminClient, DistanceMetric, Embedding, EmbeddingFunction, HnswConfig, Metadata,
    SeekDbError, ServerClient,
};
use std::sync::Arc;
mod common;
use common::load_config_for_integration;
use serde_json::json;

#[tokio::test]
async fn test_readme_doc() -> Result<(), SeekDbError> {
    let Some(config) = load_config_for_integration() else {
        return Ok(());
    };
    let client = ServerClient::from_config(config).await?;
    // 创建 admin client
    let admin = AdminClient::new(Arc::new(client.clone()));

    admin.create_database("my_test_readme", Some("sys")).await?;

    let db = admin.get_database("my_test_readme", Some("sys")).await?;
    println!("Database {:?} created successfully", db);

    let list = admin.list_databases(None, None, None).await?;
    // println!("Database list: {:?}", list);

    admin.delete_database("my_test_readme", None).await?;

    let list = admin.list_databases(None, None, None).await?;
    // println!("Database list: {:?}", list);

    // 试试 server client
    let hnsw = HnswConfig {
        dimension: 3,
        distance: DistanceMetric::Cosine,
    };
    if client.has_collection("test_readme").await? {
        client.delete_collection("test_readme").await?;
    }
    let coll = client
        .get_or_create_collection(
            "test_readme",
            Some(hnsw),
            None::<Box<dyn EmbeddingFunction>>,
        )
        .await?;

    let coll_list = client.list_collections().await?;
    println!("Collection list: {:?}", coll_list);

    let coll_test_readme = client
        .get_collection("test_readme", None::<Box<dyn EmbeddingFunction>>)
        .await?;
    println!(
        "Collection {:?} created successfully",
        coll_test_readme.name()
    );
    println!("Coll ef: {}", coll_test_readme.dimension());

    let coll_is_exit = client.has_collection("test_readme").await?;
    println!("Collection {:?} exists", coll_is_exit);

    let cnt_coll = client.count_collection().await?;
    println!(
        "Collection {:?} count: {:?}",
        coll_test_readme.name(),
        cnt_coll
    );

    // DML
    let ids = vec!["item1".to_string(), "item2".to_string()];
    let embeddings: Vec<Embedding> = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let documents = vec!["Document 1".to_string(), "Document 2".to_string()];
    let metadatas: Vec<Metadata> = vec![
        json!({"category": "AI", "score": 95}),
        json!({"category": "ML", "score": 88}),
    ];

    coll.add(&ids, Some(&embeddings), Some(&metadatas), Some(&documents))
        .await?;
    coll.update(
        &["item1".to_string()],
        Some(&vec![vec![0.7, 0.8, 0.9]]),
        Some(&vec![json!({"category": "AI", "score": 96})]),
        Some(&vec!["Updated Document 1".to_string()]),
    )
    .await?;
    let r = coll
        .get(Some(&["item1".to_string()]), None, None, None, None, None)
        .await?;
    println!("{:?}", r);

    coll.delete(
        Some(&["item1".to_string(), "item2".to_string()]),
        None,
        None,
    )
    .await?;
    assert_eq!(coll.count().await?, 0);

    Ok(())
}
