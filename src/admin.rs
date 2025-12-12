use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::server::ServerClient;
use crate::types::Database;

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

/// Thin proxy that delegates admin operations to an underlying ServerClient.
#[derive(Clone)]
pub struct AdminClient {
    inner: Arc<ServerClient>,
}

impl AdminClient {
    pub fn new(inner: Arc<ServerClient>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl AdminApi for AdminClient {
    async fn create_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        self.inner.create_database(name, tenant).await
    }

    async fn get_database(&self, name: &str, tenant: Option<&str>) -> Result<Database> {
        self.inner.get_database(name, tenant).await
    }

    async fn delete_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        self.inner.delete_database(name, tenant).await
    }

    async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<Database>> {
        self.inner.list_databases(limit, offset, tenant).await
    }
}
