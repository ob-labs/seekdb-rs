//! Unified Client trait for SeekDB
//!
//! This module provides a unified `SeekDbClient` trait that abstracts over
//! both server and embedded modes, enabling code reuse.
//!
//! Both `ServerClient` and `EmbeddedClient` implement this trait, allowing
//! high-level APIs like `Collection` to work with either mode through the
//! same interface.

use async_trait::async_trait;

use crate::backend::SqlBackend;
use crate::collection::Collection;
use crate::config::HnswConfig;
use crate::embedding::EmbeddingFunction;
use crate::error::Result;

/// Unified client trait that abstracts over server and embedded modes.
///
/// This trait provides a common interface for all SeekDB clients, allowing
/// Collection and other high-level APIs to work with both server and embedded modes.
///
/// Both `ServerClient` and `EmbeddedClient` implement this trait, enabling
/// code reuse across different connection modes.
///
/// The trait extends `SqlBackend` to provide SQL execution capabilities,
/// and adds collection management methods that are common to both modes.
///
/// # Example
///
/// ```rust,no_run
/// use seekdb_rs::{SeekDbClient, SeekDbError, ServerClient};
///
/// // Works with any client type that implements SeekDbClient (e.g. ServerClient, EmbeddedClient).
/// async fn use_client<C: SeekDbClient>(client: &C) -> Result<(), SeekDbError> {
///     client.execute("SELECT 1").await?;
///     let _collections = client.list_collections().await?;
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait SeekDbClient: SqlBackend + Send + Sync + Clone {
    /// Get the database name.
    fn database(&self) -> &str;

    /// Create a collection.
    async fn create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>>;

    /// Get an existing collection.
    async fn get_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>>;

    /// Delete a collection.
    async fn delete_collection(&self, name: &str) -> Result<()>;

    /// List all collections.
    async fn list_collections(&self) -> Result<Vec<String>>;

    /// Check if a collection exists.
    async fn has_collection(&self, name: &str) -> Result<bool>;

    /// Get or create a collection.
    async fn get_or_create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>>;

    /// Count collections.
    async fn count_collection(&self) -> Result<usize>;
}
