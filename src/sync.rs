use std::sync::Arc;

use crate::collection::Collection;
use crate::config::ServerConfig;
use crate::embedding::EmbeddingFunction;
use crate::error::{Result, SeekDbError};
use crate::filters::{DocFilter, Filter};
use crate::server::{ServerClient, ServerClientBuilder};
use crate::types::{GetResult, IncludeField, QueryResult};

/// Shared inner state for synchronous wrappers.
///
/// Holds a Tokio runtime and the underlying async `ServerClient`.
struct Inner {
    rt: tokio::runtime::Runtime,
    client: ServerClient,
}

/// Blocking/synchronous wrapper around [`ServerClient`].
///
/// This type is only available when the `sync` feature is enabled. It runs all
/// operations on an internal Tokio runtime using `block_on`.
///
/// Note: do not call these blocking APIs from within an existing Tokio runtime,
/// as that can lead to deadlocks. In async contexts, use the async
/// [`ServerClient`] APIs directly instead.
#[derive(Clone)]
pub struct SyncServerClient {
    inner: Arc<Inner>,
}

impl SyncServerClient {
    /// Build a synchronous client from a [`ServerConfig`].
    pub fn from_config(config: ServerConfig) -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SeekDbError::Other(anyhow::Error::new(e)))?
            ;
        let client = rt.block_on(ServerClient::from_config(config))?;
        let inner = Inner { rt, client };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Build a synchronous client from environment variables.
    pub fn from_env() -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SeekDbError::Other(anyhow::Error::new(e)))?
            ;
        let client = rt.block_on(ServerClient::from_env())?;
        let inner = Inner { rt, client };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Start building a [`SyncServerClient`] using a fluent builder API.
    pub fn builder() -> SyncServerClientBuilder {
        SyncServerClientBuilder::new()
    }

    /// Execute a SQL statement that does not return rows.
    pub fn execute(&self, sql: &str) -> Result<()> {
        self.inner
            .rt
            .block_on(self.inner.client.execute(sql))
            .map(|_| ())
    }

    /// Fetch all rows for the given SQL query.
    pub fn fetch_all(
        &self,
        sql: &str,
    ) -> Result<Vec<sqlx::mysql::MySqlRow>> {
        self.inner.rt.block_on(self.inner.client.fetch_all(sql))
    }

    // Collection management

    pub fn create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<crate::config::HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<SyncCollection<Ef>> {
        let collection = self.inner.rt.block_on(self.inner.client.create_collection(
            name,
            config,
            embedding_function,
        ))?;
        Ok(SyncCollection {
            inner: Arc::clone(&self.inner),
            collection,
        })
    }

    pub fn get_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        embedding_function: Option<Ef>,
    ) -> Result<SyncCollection<Ef>> {
        let collection = self
            .inner
            .rt
            .block_on(self.inner.client.get_collection(name, embedding_function))?;
        Ok(SyncCollection {
            inner: Arc::clone(&self.inner),
            collection,
        })
    }

    pub fn delete_collection(&self, name: &str) -> Result<()> {
        self.inner
            .rt
            .block_on(self.inner.client.delete_collection(name))
    }

    pub fn list_collections(&self) -> Result<Vec<String>> {
        self.inner.rt.block_on(self.inner.client.list_collections())
    }

    pub fn has_collection(&self, name: &str) -> Result<bool> {
        self.inner
            .rt
            .block_on(self.inner.client.has_collection(name))
    }

    pub fn get_or_create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<crate::config::HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<SyncCollection<Ef>> {
        let collection = self.inner.rt.block_on(
            self.inner
                .client
                .get_or_create_collection(name, config, embedding_function),
        )?;
        Ok(SyncCollection {
            inner: Arc::clone(&self.inner),
            collection,
        })
    }

    pub fn count_collection(&self) -> Result<usize> {
        self.inner.rt.block_on(self.inner.client.count_collection())
    }

    // Admin helpers

    pub fn create_database(
        &self,
        name: &str,
        tenant: Option<&str>,
    ) -> Result<()> {
        self.inner
            .rt
            .block_on(self.inner.client.create_database(name, tenant))
    }

    pub fn get_database(
        &self,
        name: &str,
        tenant: Option<&str>,
    ) -> Result<crate::types::Database> {
        self.inner
            .rt
            .block_on(self.inner.client.get_database(name, tenant))
    }

    pub fn delete_database(
        &self,
        name: &str,
        tenant: Option<&str>,
    ) -> Result<()> {
        self.inner
            .rt
            .block_on(self.inner.client.delete_database(name, tenant))
    }

    pub fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<crate::types::Database>> {
        self.inner.rt.block_on(
            self.inner
                .client
                .list_databases(limit, offset, tenant),
        )
    }
}

/// Builder for constructing a [`SyncServerClient`].
pub struct SyncServerClientBuilder {
    inner: ServerClientBuilder,
}

impl SyncServerClientBuilder {
    fn new() -> Self {
        Self {
            inner: ServerClient::builder(),
        }
    }

    /// Populate the builder from `SERVER_*` environment variables.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            inner: ServerClientBuilder::from_env()?,
        })
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.inner = self.inner.host(host);
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.inner = self.inner.port(port);
        self
    }

    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.inner = self.inner.tenant(tenant);
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.inner = self.inner.database(database);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.inner = self.inner.user(user);
        self
    }

    pub fn password(mut self, password: impl Into<String>) -> Self {
        self.inner = self.inner.password(password);
        self
    }

    pub fn max_connections(mut self, max_connections: u32) -> Self {
        self.inner = self.inner.max_connections(max_connections);
        self
    }

    /// Build a [`SyncServerClient`] using the current builder configuration.
    pub fn build(self) -> Result<SyncServerClient> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SeekDbError::Other(anyhow::Error::new(e)))?;
        let client = rt.block_on(self.inner.build())?;
        let inner = Inner { rt, client };
        Ok(SyncServerClient {
            inner: Arc::new(inner),
        })
    }
}

/// Blocking/synchronous wrapper around [`Collection`].
#[derive(Clone)]
pub struct SyncCollection<Ef = Box<dyn EmbeddingFunction>> {
    inner: Arc<Inner>,
    collection: Collection<Ef>,
}

impl<Ef: EmbeddingFunction + 'static> SyncCollection<Ef> {
    pub fn name(&self) -> &str {
        self.collection.name()
    }

    pub fn dimension(&self) -> u32 {
        self.collection.dimension()
    }

    pub fn distance(&self) -> crate::config::DistanceMetric {
        self.collection.distance()
    }

    pub fn id(&self) -> Option<&str> {
        self.collection.id()
    }

    pub fn metadata(&self) -> Option<&serde_json::Value> {
        self.collection.metadata()
    }

    pub fn add(
        &self,
        ids: &[String],
        embeddings: Option<&[crate::types::Embedding]>,
        metadatas: Option<&[crate::types::Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()> {
        self.inner
            .rt
            .block_on(self.collection.add(ids, embeddings, metadatas, documents))
    }

    pub fn update(
        &self,
        ids: &[String],
        embeddings: Option<&[crate::types::Embedding]>,
        metadatas: Option<&[crate::types::Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()> {
        self.inner.rt.block_on(self.collection.update(
            ids,
            embeddings,
            metadatas,
            documents,
        ))
    }

    pub fn upsert(
        &self,
        ids: &[String],
        embeddings: Option<&[crate::types::Embedding]>,
        metadatas: Option<&[crate::types::Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()> {
        self.inner.rt.block_on(self.collection.upsert(
            ids,
            embeddings,
            metadatas,
            documents,
        ))
    }

    pub fn delete(
        &self,
        ids: Option<&[String]>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
    ) -> Result<()> {
        self.inner
            .rt
            .block_on(self.collection.delete(ids, where_meta, where_doc))
    }

    pub fn query_embeddings(
        &self,
        embeddings: &[crate::types::Embedding],
        n_results: u32,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        self.inner.rt.block_on(self.collection.query_embeddings(
            embeddings,
            n_results,
            where_meta,
            where_doc,
            include,
        ))
    }

    pub fn query_texts(
        &self,
        texts: &[String],
        n_results: u32,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        self.inner.rt.block_on(self.collection.query_texts(
            texts,
            n_results,
            where_meta,
            where_doc,
            include,
        ))
    }

    pub fn hybrid_search(
        &self,
        queries: &[String],
        search_params: Option<&serde_json::Value>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        n_results: u32,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        self.inner.rt.block_on(self.collection.hybrid_search(
            queries,
            search_params,
            where_meta,
            where_doc,
            n_results,
            include,
        ))
    }

    pub fn hybrid_search_advanced(
        &self,
        query: Option<crate::collection::HybridQuery>,
        knn: Option<crate::collection::HybridKnn>,
        rank: Option<crate::collection::HybridRank>,
        n_results: u32,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        self.inner.rt.block_on(self.collection.hybrid_search_advanced(
            query,
            knn,
            rank,
            n_results,
            include,
        ))
    }

    pub fn get(
        &self,
        ids: Option<&[String]>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        limit: Option<u32>,
        offset: Option<u32>,
        include: Option<&[IncludeField]>,
    ) -> Result<GetResult> {
        self.inner.rt.block_on(self.collection.get(
            ids,
            where_meta,
            where_doc,
            limit,
            offset,
            include,
        ))
    }

    pub fn count(&self) -> Result<u64> {
        self.inner.rt.block_on(self.collection.count())
    }

    pub fn peek(&self, limit: u32) -> Result<GetResult> {
        self.inner.rt.block_on(self.collection.peek(limit))
    }
}
