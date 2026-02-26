//! Integration tests for embedded mode with DefaultEmbedding.
#![cfg_attr(not(all(feature = "embedded", feature = "embedding")), allow(dead_code))]

#[cfg(all(feature = "embedded", feature = "embedding"))]
#[path = "embedded/common.rs"]
mod common;

#[cfg(not(all(feature = "embedded", feature = "embedding")))]
fn main() {}

#[cfg(all(feature = "embedded", feature = "embedding"))]
mod run {
    use anyhow::Result;
    use seekdb_rs::{
        AddBatch, Client, DefaultEmbedding, DistanceMetric, EmbeddingFunction, GetQuery, HnswConfig,
        IncludeField,
    };

    use crate::common::{run_embedded_tests, shared_db_dir, ts_suffix};

    pub fn main() {
        run_embedded_tests(run_tests);
    }

    async fn run_tests() -> Result<()> {
        default_embedding_new().await?;
        collection_add_with_auto_embedding_default_embedding().await?;
        collection_query_texts_with_default_embedding().await?;
        Ok(())
    }

    /// DefaultEmbedding::new() and dimension (mirrors server collection_*_default_embedding tests).
    async fn default_embedding_new() -> Result<()> {
        let db_dir = shared_db_dir();
        let _client = Client::builder()
            .path(db_dir.to_string_lossy().as_ref())
            .database("test")
            .build()
            .await?;
        let ef = DefaultEmbedding::new()?;
        assert!(ef.dimension() > 0);
        Ok(())
    }

    async fn collection_add_with_auto_embedding_default_embedding() -> Result<()> {
        let db_dir = shared_db_dir();
        let client = Client::builder()
            .path(db_dir.to_string_lossy().as_ref())
            .database("test")
            .build()
            .await?;
        let coll_name = format!("auto_onnx_coll_{}", ts_suffix());
        let ef = DefaultEmbedding::new()?;
        let dim = ef.dimension() as u32;
        let hnsw = HnswConfig {
            dimension: dim,
            distance: DistanceMetric::Cosine,
        };
        let coll = client
            .create_collection::<DefaultEmbedding>(&coll_name, Some(hnsw), Some(ef))
            .await?;
        let ids = vec!["onnx1".to_string(), "onnx2".to_string()];
        let docs = vec![
            "seekdb rust integration".to_string(),
            "vector search with onnx".to_string(),
        ];
        coll.add_batch(AddBatch::new(&ids).documents(&docs)).await?;
        let got = coll
            .get_query(
                GetQuery::new().with_include(&[
                    IncludeField::Documents,
                    IncludeField::Metadatas,
                    IncludeField::Embeddings,
                ]),
            )
            .await?;
        assert_eq!(got.ids.len(), ids.len());
        assert!(got.documents.as_ref().map(|d| d.len() == ids.len()).unwrap_or(false));
        assert!(got.embeddings.as_ref().map(|e| e.len() == ids.len()).unwrap_or(false));
        client.delete_collection(&coll_name).await.ok();
        Ok(())
    }

    async fn collection_query_texts_with_default_embedding() -> Result<()> {
        let db_dir = shared_db_dir();
        let client = Client::builder()
            .path(db_dir.to_string_lossy().as_ref())
            .database("test")
            .build()
            .await?;
        let coll_name = format!("qtexts_onnx_coll_{}", ts_suffix());
        let ef = DefaultEmbedding::new()?;
        let dim = ef.dimension() as u32;
        let hnsw = HnswConfig {
            dimension: dim,
            distance: DistanceMetric::Cosine,
        };
        let coll = client
            .create_collection::<DefaultEmbedding>(&coll_name, Some(hnsw), Some(ef))
            .await?;
        let ids = vec!["qt_onnx1".to_string(), "qt_onnx2".to_string()];
        let docs = vec![
            "hello rust with onnx".to_string(),
            "hello seekdb embeddings".to_string(),
        ];
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
}

#[cfg(all(feature = "embedded", feature = "embedding"))]
fn main() {
    run::main();
}
