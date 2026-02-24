#[cfg(feature = "server")]
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
#[cfg(all(feature = "server", not(feature = "embedded")))]
use crate::error::SeekDbError;
use crate::types::Database;

#[cfg(feature = "server")]
use crate::server::ServerClient;

#[cfg(feature = "embedded")]
use crate::embedded::EmbeddedClient;

/// Bootstrap database name for admin operations (server mode).
/// Align with pyseekdb AdminClient using `information_schema`.
pub const ADMIN_BOOTSTRAP_DATABASE: &str = "information_schema";

/// Admin API for database management
#[async_trait]
pub trait AdminApi {
    async fn create_database(&self, name: &str, tenant: Option<&str>) -> Result<()>;
    async fn get_database(&self, name: &str, tenant: Option<&str>) -> Result<Database>;
    async fn delete_database(&self, name: &str, tenant: Option<&str>) -> Result<()>;
    async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<Database>>;
}

/// Unified admin client: server or embedded. Uses bootstrap DB (information_schema), aligned with pyseekdb.
/// Align with pyseekdb's `AdminClient()` factory returning `_AdminClientProxy`.
#[derive(Clone)]
pub enum AdminClient {
    #[cfg(feature = "server")]
    Server(ServerClient),
    #[cfg(feature = "embedded")]
    Embedded(EmbeddedClient),
}

impl AdminClient {
    /// Create an admin client from a server client (e.g. connected to information_schema).
    #[cfg(feature = "server")]
    pub fn from_server(client: ServerClient) -> Self {
        AdminClient::Server(client)
    }

    /// Create an admin client from an embedded client (e.g. connected to information_schema for admin).
    #[cfg(feature = "embedded")]
    pub fn from_embedded(client: EmbeddedClient) -> Self {
        AdminClient::Embedded(client)
    }

    /// Legacy: create from `Arc<ServerClient>`. Prefer `AdminClient::builder()` or `from_server`.
    #[cfg(feature = "server")]
    pub fn new(inner: Arc<ServerClient>) -> Self {
        AdminClient::Server((*inner).clone())
    }

    /// Create a builder for unified admin client (path => embedded, host => server).
    pub fn builder() -> AdminClientBuilder {
        AdminClientBuilder::new()
    }
}

/// Builder for unified [`AdminClient`]. Uses bootstrap DB for admin ops.
pub struct AdminClientBuilder {
    #[cfg(feature = "embedded")]
    path: Option<String>,
    #[cfg(feature = "server")]
    host: Option<String>,
    #[cfg(feature = "server")]
    port: u16,
    #[cfg(feature = "server")]
    tenant: String,
    #[cfg(feature = "server")]
    user: String,
    #[cfg(feature = "server")]
    password: String,
    #[cfg(feature = "embedded")]
    autocommit: bool,
    #[cfg(feature = "embedded")]
    port_embedded: Option<i32>,
    #[cfg(feature = "embedded")]
    skip_open: bool,
    #[cfg(feature = "server")]
    max_connections: u32,
}

impl AdminClientBuilder {
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
            #[cfg(feature = "server")]
            user: "root".to_string(),
            #[cfg(feature = "server")]
            password: String::new(),
            #[cfg(feature = "embedded")]
            autocommit: false,
            #[cfg(feature = "embedded")]
            port_embedded: None,
            #[cfg(feature = "embedded")]
            skip_open: false,
            #[cfg(feature = "server")]
            max_connections: 5,
        }
    }

    #[cfg(feature = "embedded")]
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

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
    pub fn skip_open(mut self, skip_open: bool) -> Self {
        self.skip_open = skip_open;
        self
    }

    #[cfg(all(feature = "server", feature = "embedded"))]
    pub async fn build(self) -> Result<AdminClient> {
        if let Some(path) = self.path {
            let client = EmbeddedClient::builder()
                .db_dir(path)
                .database(ADMIN_BOOTSTRAP_DATABASE) // same as server: information_schema for admin ops
                .autocommit(self.autocommit)
                .port(self.port_embedded)
                .skip_open(self.skip_open)
                .build()
                .await?;
            Ok(AdminClient::Embedded(client))
        } else if let Some(host) = self.host {
            let client = ServerClient::builder()
                .host(host)
                .port(self.port)
                .tenant(self.tenant)
                .database(ADMIN_BOOTSTRAP_DATABASE)
                .user(self.user)
                .password(self.password)
                .max_connections(self.max_connections)
                .build()
                .await?;
            Ok(AdminClient::Server(client))
        } else {
            let default_path = "seekdb.db".to_string();
            let client = EmbeddedClient::builder()
                .db_dir(default_path)
                .database(ADMIN_BOOTSTRAP_DATABASE)
                .autocommit(self.autocommit)
                .port(self.port_embedded)
                .skip_open(self.skip_open)
                .build()
                .await?;
            Ok(AdminClient::Embedded(client))
        }
    }

    #[cfg(all(feature = "server", not(feature = "embedded")))]
    pub async fn build(self) -> Result<AdminClient> {
        if let Some(host) = self.host {
            let client = ServerClient::builder()
                .host(host)
                .port(self.port)
                .tenant(self.tenant)
                .database(ADMIN_BOOTSTRAP_DATABASE)
                .user(self.user)
                .password(self.password)
                .max_connections(self.max_connections)
                .build()
                .await?;
            Ok(AdminClient::Server(client))
        } else {
            Err(SeekDbError::Config(
                "either path (embedded) or host (server) must be set".into(),
            ))
        }
    }

    #[cfg(all(not(feature = "server"), feature = "embedded"))]
    pub async fn build(self) -> Result<AdminClient> {
        let path = self.path.unwrap_or_else(|| "seekdb.db".to_string());
        let client = EmbeddedClient::builder()
            .db_dir(path)
            .database(ADMIN_BOOTSTRAP_DATABASE)
            .autocommit(self.autocommit)
            .port(self.port_embedded)
            .skip_open(self.skip_open)
            .build()
            .await?;
        Ok(AdminClient::Embedded(client))
    }
}

impl Default for AdminClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AdminApi for AdminClient {
    async fn create_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        match self {
            #[cfg(feature = "server")]
            AdminClient::Server(c) => c.create_database(name, tenant).await,
            #[cfg(feature = "embedded")]
            AdminClient::Embedded(c) => c.create_database(name, tenant).await,
        }
    }

    async fn get_database(&self, name: &str, tenant: Option<&str>) -> Result<Database> {
        match self {
            #[cfg(feature = "server")]
            AdminClient::Server(c) => c.get_database(name, tenant).await,
            #[cfg(feature = "embedded")]
            AdminClient::Embedded(c) => c.get_database(name, tenant).await,
        }
    }

    async fn delete_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        match self {
            #[cfg(feature = "server")]
            AdminClient::Server(c) => c.delete_database(name, tenant).await,
            #[cfg(feature = "embedded")]
            AdminClient::Embedded(c) => c.delete_database(name, tenant).await,
        }
    }

    async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<Database>> {
        match self {
            #[cfg(feature = "server")]
            AdminClient::Server(c) => c.list_databases(limit, offset, tenant).await,
            #[cfg(feature = "embedded")]
            AdminClient::Embedded(c) => c.list_databases(limit, offset, tenant).await,
        }
    }
}
