//! Unified Client interface for SeekDB
//!
//! This module provides a unified `Client` enum and `ClientBuilder` that
//! select embedded or server mode based on parameters (path vs host),
//! aligned with pyseekdb's `Client()` factory.

use async_trait::async_trait;

use crate::admin::AdminApi;
use crate::backend::{BackendRow, CollectionBackend, QueryParam};
use crate::collection::Collection;
use crate::config::HnswConfig;
#[cfg(feature = "server")]
use crate::config::ServerConfig;
#[cfg(feature = "embedded")]
use crate::config::EmbeddedConfig;
use crate::embedding::EmbeddingFunction;
use crate::error::Result;
#[cfg(all(feature = "server", not(feature = "embedded")))]
use crate::error::SeekDbError;

#[cfg(feature = "server")]
use crate::server::ServerClient;

#[cfg(feature = "embedded")]
use crate::embedded::EmbeddedClient;

/// Unified client that can be either a server client or an embedded client.
#[derive(Clone)]
pub enum Client {
    #[cfg(feature = "server")]
    Server(ServerClient),
    #[cfg(feature = "embedded")]
    Embedded(EmbeddedClient),
}

/// Builder for unified [`Client`]. Use `path` for embedded mode, or `host`/`port` for server mode.
///
/// Aligned with pyseekdb's `Client(path=..., database=...)` / `Client(host=..., port=..., database=...)`.
pub struct ClientBuilder {
    #[cfg(feature = "embedded")]
    path: Option<String>,
    #[cfg(feature = "server")]
    host: Option<String>,
    #[cfg(feature = "server")]
    port: u16,
    #[cfg(feature = "server")]
    tenant: String,
    database: String,
    #[cfg(feature = "server")]
    user: String,
    #[cfg(feature = "server")]
    password: String,
    #[cfg(feature = "embedded")]
    autocommit: bool,
    #[cfg(feature = "embedded")]
    port_embedded: Option<i32>,
    #[cfg(feature = "server")]
    max_connections: u32,
}

impl ClientBuilder {
    /// Create a new builder with defaults (database = "test", server defaults for host/port/tenant/user).
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "embedded")]
            path: None,
            #[cfg(feature = "server")]
            host: None,
            #[cfg(feature = "server")]
            port: 2881,
            #[cfg(feature = "server")]
            tenant: "sys".to_string(),
            database: "test".to_string(),
            #[cfg(feature = "server")]
            user: "root".to_string(),
            #[cfg(feature = "server")]
            password: String::new(),
            #[cfg(feature = "embedded")]
            autocommit: false,
            #[cfg(feature = "embedded")]
            port_embedded: None,
            #[cfg(feature = "server")]
            max_connections: 5,
        }
    }

    /// Set path (embedded mode). When set, build() will create an embedded client.
    #[cfg(feature = "embedded")]
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set host (server mode). When set, build() will create a server client.
    #[cfg(feature = "server")]
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    #[cfg(feature = "server")]
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    #[cfg(feature = "server")]
    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = tenant.into();
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    #[cfg(feature = "server")]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = user.into();
        self
    }

    #[cfg(feature = "server")]
    pub fn password(mut self, password: impl Into<String>) -> Self {
        self.password = password.into();
        self
    }

    #[cfg(feature = "embedded")]
    pub fn autocommit(mut self, autocommit: bool) -> Self {
        self.autocommit = autocommit;
        self
    }

    #[cfg(feature = "embedded")]
    pub fn port_embedded(mut self, port: Option<i32>) -> Self {
        self.port_embedded = port;
        self
    }

    #[cfg(feature = "server")]
    pub fn max_connections(mut self, max_connections: u32) -> Self {
        self.max_connections = max_connections;
        self
    }

    /// Create a builder from embedded config (path + database). Align with pyseekdb Client(path=..., database=...).
    #[cfg(feature = "embedded")]
    pub fn from_embedded_config(config: EmbeddedConfig) -> Self {
        Self::new()
            .path(config.db_dir)
            .database(config.database)
            .autocommit(config.autocommit)
            .port_embedded(config.port)
    }

    /// Create a builder from server config (host + port + database). Align with pyseekdb Client(host=..., port=..., database=...).
    #[cfg(feature = "server")]
    pub fn from_server_config(config: ServerConfig) -> Self {
        Self::new()
            .host(config.host)
            .port(config.port)
            .tenant(config.tenant)
            .database(config.database)
            .user(config.user)
            .password(config.password)
            .max_connections(config.max_connections)
    }

    /// Build the unified [`Client`]. Uses embedded if `path` was set, otherwise server if `host` was set.
    #[cfg(all(feature = "server", feature = "embedded"))]
    pub async fn build(self) -> Result<Client> {
        if let Some(path) = self.path {
            let b = EmbeddedClient::builder()
                .db_dir(path)
                .database(self.database)
                .autocommit(self.autocommit)
                .port(self.port_embedded);
            let client = b.build().await?;
            Ok(Client::Embedded(client))
        } else if let Some(host) = self.host {
            let client = ServerClient::builder()
                .host(host)
                .port(self.port)
                .tenant(self.tenant)
                .database(self.database)
                .user(self.user)
                .password(self.password)
                .max_connections(self.max_connections)
                .build()
                .await?;
            Ok(Client::Server(client))
        } else {
            // Default: embedded with default path (align with pyseekdb: seekdb.db)
            let default_path = "seekdb.db".to_string();
            let b = EmbeddedClient::builder()
                .db_dir(default_path)
                .database(self.database)
                .autocommit(self.autocommit)
                .port(self.port_embedded);
            let client = b.build().await?;
            Ok(Client::Embedded(client))
        }
    }

    #[cfg(all(feature = "server", not(feature = "embedded")))]
    pub async fn build(self) -> Result<Client> {
        if let Some(host) = self.host {
            let client = ServerClient::builder()
                .host(host)
                .port(self.port)
                .tenant(self.tenant)
                .database(self.database)
                .user(self.user)
                .password(self.password)
                .max_connections(self.max_connections)
                .build()
                .await?;
            Ok(Client::Server(client))
        } else {
            Err(SeekDbError::Config(
                "either path (embedded) or host (server) must be set".into(),
            ))
        }
    }

    #[cfg(all(not(feature = "server"), feature = "embedded"))]
    pub async fn build(self) -> Result<Client> {
        let path = self.path.unwrap_or_else(|| "seekdb.db".to_string());
        let b = EmbeddedClient::builder()
            .db_dir(path)
            .database(self.database)
            .autocommit(self.autocommit)
            .port(self.port_embedded);
        let client = b.build().await?;
        Ok(Client::Embedded(client))
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Create a unified client builder (path => embedded, host => server).
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Return the current database name.
    pub fn database(&self) -> &str {
        match self {
            #[cfg(feature = "server")]
            Client::Server(c) => c.database(),
            #[cfg(feature = "embedded")]
            Client::Embedded(c) => c.database(),
        }
    }

    /// Execute a SQL statement that does not return rows.
    /// If `params` is `Some` with non-empty slice, uses parameterized execution (`?` placeholders in `sql`).
    /// If `params` is `None` or `Some(&[])`, executes `sql` as-is (no parameters).
    pub async fn execute(&self, sql: &str, params: Option<&[QueryParam]>) -> Result<()> {
        let use_params = params.map(|p| !p.is_empty()).unwrap_or(false);
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => {
                if use_params {
                    CollectionBackend::execute_with_params(client, sql, params.unwrap()).await
                } else {
                    client.execute(sql).await.map(|_| ())
                }
            }
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => {
                if use_params {
                    CollectionBackend::execute_with_params(client, sql, params.unwrap()).await
                } else {
                    client.execute(sql).await
                }
            }
        }
    }

    /// Execute a query and return all rows (unified BackendRow).
    /// If `params` is `Some` with non-empty slice, uses parameterized execution (`?` placeholders in `sql`).
    /// If `params` is `None` or `Some(&[])`, executes `sql` as-is (no parameters).
    pub async fn fetch_all(&self, sql: &str, params: Option<&[QueryParam]>) -> Result<Vec<Box<dyn BackendRow>>> {
        let use_params = params.map(|p| !p.is_empty()).unwrap_or(false);
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => {
                if use_params {
                    CollectionBackend::fetch_all_with_params(client, sql, params.unwrap()).await
                } else {
                    CollectionBackend::fetch_all(client, sql).await
                }
            }
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => {
                if use_params {
                    CollectionBackend::fetch_all_with_params(client, sql, params.unwrap()).await
                } else {
                    CollectionBackend::fetch_all(client, sql).await
                }
            }
        }
    }

    /// Create a collection (delegates to the underlying client).
    pub async fn create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.create_collection(name, config, embedding_function).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.create_collection(name, config, embedding_function).await,
        }
    }

    /// Get an existing collection.
    pub async fn get_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.get_collection(name, embedding_function).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.get_collection(name, embedding_function).await,
        }
    }

    /// Delete a collection.
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.delete_collection(name).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.delete_collection(name).await,
        }
    }

    /// List all collection names in the current database.
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.list_collections().await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.list_collections().await,
        }
    }

    /// Check if a collection exists.
    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.has_collection(name).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.has_collection(name).await,
        }
    }

    /// Get or create a collection.
    pub async fn get_or_create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.get_or_create_collection(name, config, embedding_function).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.get_or_create_collection(name, config, embedding_function).await,
        }
    }

    /// Count collections in the current database.
    pub async fn count_collection(&self) -> Result<usize> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.count_collection().await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.count_collection().await,
        }
    }
}

// Note: SqlBackend implementation for Client is complex due to different Row types
// For now, use the concrete client types (ServerClient or EmbeddedClient) directly
// A unified Client can be added later with proper type erasure or generics

#[async_trait]
impl AdminApi for Client {
    async fn create_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.create_database(name, tenant).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.create_database(name, tenant).await,
        }
    }

    async fn get_database(&self, name: &str, tenant: Option<&str>) -> Result<crate::types::Database> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.get_database(name, tenant).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.get_database(name, tenant).await,
        }
    }

    async fn delete_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.delete_database(name, tenant).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.delete_database(name, tenant).await,
        }
    }

    async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<crate::types::Database>> {
        match self {
            #[cfg(feature = "server")]
            Client::Server(client) => client.list_databases(limit, offset, tenant).await,
            #[cfg(feature = "embedded")]
            Client::Embedded(client) => client.list_databases(limit, offset, tenant).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Server-only build: builder without host (and no path) must return Config error.
    #[cfg(all(feature = "server", not(feature = "embedded")))]
    #[tokio::test]
    async fn client_builder_fails_without_host_or_path() {
        let res = Client::builder().build().await;
        match res {
            Err(crate::error::SeekDbError::Config(_)) => {}
            Ok(_) => panic!("expected Err(SeekDbError::Config)"),
            Err(_) => panic!("expected SeekDbError::Config variant"),
        }
    }

    /// Embedded build: empty path is rejected by EmbeddedClient.
    #[cfg(feature = "embedded")]
    #[tokio::test]
    async fn client_builder_embedded_rejects_empty_path() {
        let res = Client::builder().path("").database("test").build().await;
        assert!(res.is_err());
    }

    /// ClientBuilder implements Default and is constructible.
    #[test]
    fn client_builder_default_and_new() {
        let _ = ClientBuilder::new();
        let _ = ClientBuilder::default();
    }
}
