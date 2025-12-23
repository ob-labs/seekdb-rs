use async_trait::async_trait;
use sqlx::mysql::MySqlPoolOptions;
use sqlx::{MySqlPool, Row};

use crate::admin::AdminApi;
use crate::backend::SqlBackend;
use crate::collection::Collection;
use crate::config::{DistanceMetric, HnswConfig, ServerConfig};
use crate::embedding::EmbeddingFunction;
use crate::error::{Result, SeekDbError};
use crate::meta::CollectionNames;
use crate::types::Database;

/// Builder for configuring and constructing a [`ServerClient`].
///
/// This provides a more ergonomic, chainable way to configure connection
/// parameters, mirroring the `ServerConfig` structure while keeping existing
/// `ServerClient::from_config` / `from_env` APIs intact.
pub struct ServerClientBuilder {
    host: String,
    port: u16,
    tenant: String,
    database: String,
    user: String,
    password: String,
    max_connections: u32,
}

/// Server-side client that talks to seekdb/OceanBase over MySQL protocol.
#[derive(Clone)]
pub struct ServerClient {
    pool: MySqlPool,
    tenant: String,
    database: String,
}

impl ServerClient {
    /// Build a client from a `ServerConfig`.
    pub async fn from_config(config: ServerConfig) -> Result<Self> {
        Self::connect_internal(
            &config.host,
            config.port,
            &config.tenant,
            &config.database,
            &config.user,
            &config.password,
            config.max_connections,
        )
        .await
    }

    pub async fn from_env() -> Result<Self> {
        let config = ServerConfig::from_env()?;
        Self::from_config(config).await
    }

    pub fn pool(&self) -> &MySqlPool {
        &self.pool
    }

    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    pub fn database(&self) -> &str {
        &self.database
    }

    pub fn builder() -> ServerClientBuilder {
        ServerClientBuilder::new()
    }

    /// Execute a SQL statement that does not return rows.
    pub async fn execute(&self, sql: &str) -> Result<sqlx::mysql::MySqlQueryResult> {
        sqlx::query(sql)
            .execute(&self.pool)
            .await
            .map_err(Into::into)
    }

    /// Fetch all rows for the given SQL query.
    pub async fn fetch_all(&self, sql: &str) -> Result<Vec<sqlx::mysql::MySqlRow>> {
        sqlx::query(sql)
            .fetch_all(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        let cfg = config.ok_or_else(|| {
            SeekDbError::Config("HnswConfig must be provided when creating a collection".into())
        })?;

        let table_name = CollectionNames::table_name(name);
        let sql = build_create_table_sql(&table_name, cfg.dimension, cfg.distance);
        self.execute(&sql).await?;

        Ok(Collection::new(
            std::sync::Arc::new(self.clone()),
            name.to_string(),
            None,
            cfg.dimension,
            cfg.distance,
            embedding_function,
            None,
        ))
    }

    pub async fn get_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        let table_name = CollectionNames::table_name(name);

        // Check existence by describing the table
        let describe_sql = format!("DESCRIBE `{table_name}`");
        let describe = self.fetch_all(&describe_sql).await?;
        if describe.is_empty() {
            return Err(SeekDbError::NotFound(format!(
                "collection not found: {name}"
            )));
        }

        // Extract dimension from embedding column type
        let mut dimension: Option<u32> = None;
        for row in describe {
            let field: String = row.try_get("Field").unwrap_or_default();
            if field == "embedding" {
                let type_str: String = row.try_get("Type").unwrap_or_default();
                if let Some(dim) = parse_dimension(&type_str) {
                    dimension = Some(dim);
                }
                break;
            }
        }

        // Extract distance from SHOW CREATE TABLE
        let create_sql = format!("SHOW CREATE TABLE `{table_name}`");
        let create_rows = self.fetch_all(&create_sql).await?;
        let mut distance: DistanceMetric = DistanceMetric::L2;
        if let Some(row) = create_rows.first() {
            let create_stmt: String = row
                .try_get("Create Table")
                .or_else(|_| row.try_get(1))
                .unwrap_or_default();
            if let Some(d) = parse_distance(&create_stmt) {
                distance = d;
            }
        }

        let dimension = dimension.ok_or_else(|| {
            SeekDbError::Config("cannot detect dimension from collection schema".into())
        })?;

        Ok(Collection::new(
            std::sync::Arc::new(self.clone()),
            name.to_string(),
            None,
            dimension,
            distance,
            embedding_function,
            None,
        ))
    }

    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        let table_name = CollectionNames::table_name(name);
        let sql = format!("DROP TABLE IF EXISTS `{table_name}`");
        self.execute(&sql).await?;
        Ok(())
    }

    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let rows = match self.fetch_all("SHOW TABLES LIKE 'c$v1$%'").await {
            Ok(rows) => rows,
            Err(_) => {
                // Fallback to information_schema if SHOW TABLES is not supported
                let sql = format!(
                    "SELECT TABLE_NAME FROM information_schema.TABLES \
                     WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME LIKE 'c$v1$%'",
                    self.database
                );
                self.fetch_all(&sql).await?
            }
        };

        let mut names = Vec::new();
        for row in rows {
            // SHOW TABLES column name varies; take first column
            if let Ok(table_name) = row.try_get::<String, _>(0) {
                if let Some(name) = table_name.strip_prefix("c$v1$") {
                    names.push(name.to_string());
                }
            }
        }
        Ok(names)
    }

    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        let table_name = CollectionNames::table_name(name);
        let sql = format!(
            "SELECT 1 FROM information_schema.TABLES \
             WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? LIMIT 1"
        );
        let exists = sqlx::query(&sql)
            .bind(&self.database)
            .bind(&table_name)
            .fetch_optional(&self.pool)
            .await?;
        Ok(exists.is_some())
    }

    /// Convenience: get if exists, else create.
    pub async fn get_or_create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        if self.has_collection(name).await? {
            self.get_collection(name, embedding_function).await
        } else {
            self.create_collection(name, config, embedding_function)
                .await
        }
    }

    pub async fn count_collection(&self) -> Result<usize> {
        let collections = self.list_collections().await?;
        Ok(collections.len())
    }

    // ---- Internal admin helpers (shared by inherent & trait impl) ----
    async fn create_database_impl(&self, _name: &str, _tenant: Option<&str>) -> Result<()> {
        let sql = format!("CREATE DATABASE IF NOT EXISTS {}", escape_identifier(_name));
        self.execute(&sql).await?;
        Ok(())
    }

    async fn get_database_impl(&self, _name: &str, _tenant: Option<&str>) -> Result<Database> {
        let tenant = self.effective_tenant(_tenant).to_string();
        let row = sqlx::query(
            "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME \
             FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = ?",
        )
        .bind(_name)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else {
            return Err(SeekDbError::NotFound(format!(
                "database not found: {_name}"
            )));
        };

        Ok(Database {
            name: row.try_get::<String, _>("SCHEMA_NAME")?,
            tenant: Some(tenant),
            charset: row.try_get("DEFAULT_CHARACTER_SET_NAME").ok(),
            collation: row.try_get("DEFAULT_COLLATION_NAME").ok(),
        })
    }

    async fn delete_database_impl(&self, _name: &str, _tenant: Option<&str>) -> Result<()> {
        let sql = format!("DROP DATABASE IF EXISTS {}", escape_identifier(_name));
        self.execute(&sql).await?;
        Ok(())
    }

    async fn list_databases_impl(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        _tenant: Option<&str>,
    ) -> Result<Vec<Database>> {
        let mut sql = String::from(
            "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME \
             FROM information_schema.SCHEMATA",
        );

        if let Some(limit) = limit {
            sql.push_str(&format!(" LIMIT {limit}"));
        }
        if let Some(offset) = offset {
            // MySQL allows OFFSET only when LIMIT exists; use a large limit when missing.
            if limit.is_none() {
                sql.push_str(" LIMIT 18446744073709551615");
            }
            sql.push_str(&format!(" OFFSET {offset}"));
        }

        let tenant = self.effective_tenant(_tenant).to_string();
        let rows = self.fetch_all(&sql).await?;

        let mut databases = Vec::with_capacity(rows.len());
        for row in rows {
            databases.push(Database {
                name: row.try_get("SCHEMA_NAME")?,
                tenant: Some(tenant.clone()),
                charset: row.try_get("DEFAULT_CHARACTER_SET_NAME").ok(),
                collation: row.try_get("DEFAULT_COLLATION_NAME").ok(),
            });
        }

        Ok(databases)
    }

    // Optional ergonomic inherent methods matching AdminApi for direct calls.
    pub async fn create_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        self.create_database_impl(name, tenant).await
    }

    pub async fn get_database(&self, name: &str, tenant: Option<&str>) -> Result<Database> {
        self.get_database_impl(name, tenant).await
    }

    pub async fn delete_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        self.delete_database_impl(name, tenant).await
    }

    pub async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<Database>> {
        self.list_databases_impl(limit, offset, tenant).await
    }
}

#[async_trait]
impl AdminApi for ServerClient {
    async fn create_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        self.create_database_impl(name, tenant).await
    }

    async fn get_database(&self, name: &str, tenant: Option<&str>) -> Result<Database> {
        self.get_database_impl(name, tenant).await
    }

    async fn delete_database(&self, name: &str, tenant: Option<&str>) -> Result<()> {
        self.delete_database_impl(name, tenant).await
    }

    async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<Database>> {
        self.list_databases_impl(limit, offset, tenant).await
    }
}

// Implement the generic SqlBackend abstraction for ServerClient so that
// higher-level code can depend on SqlBackend instead of this concrete type.
#[async_trait]
impl SqlBackend for ServerClient {
    type Row = sqlx::mysql::MySqlRow;

    async fn execute(&self, sql: &str) -> crate::error::Result<()> {
        // Delegate to the inherent execute and discard the driver-specific result.
        ServerClient::execute(self, sql).await.map(|_| ())
    }

    async fn fetch_all(&self, sql: &str) -> crate::error::Result<Vec<Self::Row>> {
        ServerClient::fetch_all(self, sql).await
    }

    fn mode(&self) -> &'static str {
        "server"
    }
}

impl ServerClient {
    fn effective_tenant<'a>(&'a self, tenant: Option<&'a str>) -> &'a str {
        tenant.unwrap_or(&self.tenant)
    }
}

fn build_create_table_sql(table_name: &str, dimension: u32, distance: DistanceMetric) -> String {
    let distance = distance_str(distance);
    format!(
        "CREATE TABLE `{table_name}` (
            _id varbinary(512) PRIMARY KEY NOT NULL,
            document text,
            embedding vector({dimension}),
            metadata json,
            FULLTEXT INDEX idx_fts(document) WITH PARSER ik,
            VECTOR INDEX idx_vec (embedding) with(distance={distance}, type=hnsw, lib=vsag)
        ) ORGANIZATION = HEAP;"
    )
}

fn distance_str(distance: DistanceMetric) -> &'static str {
    match distance {
        DistanceMetric::L2 => "l2",
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::InnerProduct => "inner_product",
    }
}

fn escape_identifier(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

fn connect_url(
    host: &str,
    port: u16,
    tenant: &str,
    database: &str,
    user: &str,
    password: &str,
) -> String {
    let user_tenant = format!("{user}@{tenant}");
    format!("mysql://{user_tenant}:{password}@{host}:{port}/{database}")
}

impl ServerClient {
    async fn connect_internal(
        host: &str,
        port: u16,
        tenant: &str,
        database: &str,
        user: &str,
        password: &str,
        max_connections: u32,
    ) -> Result<Self> {
        let url = connect_url(host, port, tenant, database, user, password);
        let pool = MySqlPoolOptions::new()
            .max_connections(max_connections)
            .connect(&url)
            .await
            .map_err(|e| SeekDbError::Connection(e.to_string()))?;

        Ok(Self {
            pool,
            tenant: tenant.to_string(),
            database: database.to_string(),
        })
    }
}

impl ServerClientBuilder {
    fn new() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 2881,
            tenant: "sys".to_string(),
            database: "test".to_string(),
            user: "root".to_string(),
            password: String::new(),
            max_connections: 5,
        }
    }

    /// Populate the builder from `SERVER_*` environment variables using
    /// [`ServerConfig::from_env`]. Individual fields can still be overridden
    /// afterwards via the other builder methods.
    pub fn from_env() -> Result<Self> {
        let config = ServerConfig::from_env()?;
        Ok(Self {
            host: config.host,
            port: config.port,
            tenant: config.tenant,
            database: config.database,
            user: config.user,
            password: config.password,
            max_connections: config.max_connections,
        })
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = tenant.into();
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = user.into();
        self
    }

    pub fn password(mut self, password: impl Into<String>) -> Self {
        self.password = password.into();
        self
    }

    pub fn max_connections(mut self, max_connections: u32) -> Self {
        self.max_connections = max_connections;
        self
    }

    /// Build a [`ServerClient`] using the current builder configuration.
    pub async fn build(self) -> Result<ServerClient> {
        ServerClient::connect_internal(
            &self.host,
            self.port,
            &self.tenant,
            &self.database,
            &self.user,
            &self.password,
            self.max_connections,
        )
        .await
    }
}

fn parse_dimension(type_str: &str) -> Option<u32> {
    // expect something like "vector(384)"
    let lower = type_str.to_lowercase();
    if let Some(start) = lower.find("vector(") {
        let rest = &lower[start + "vector(".len()..];
        if let Some(end) = rest.find(')') {
            if let Ok(dim) = rest[..end].trim().parse::<u32>() {
                return Some(dim);
            }
        }
    }
    None
}

fn parse_distance(create_stmt: &str) -> Option<DistanceMetric> {
    // look for "distance=<value>" inside the create table statement
    let lower = create_stmt.to_lowercase();
    if let Some(pos) = lower.find("distance=") {
        let rest = &lower[pos + "distance=".len()..];
        let value: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        return match value.as_str() {
            "l2" => Some(DistanceMetric::L2),
            "cosine" => Some(DistanceMetric::Cosine),
            "inner_product" | "ip" => Some(DistanceMetric::InnerProduct),
            _ => None,
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dimension() {
        assert_eq!(parse_dimension("vector(384)"), Some(384));
        assert_eq!(parse_dimension("VECTOR(128)"), Some(128));
        assert_eq!(parse_dimension("text"), None);
    }

    #[test]
    fn test_parse_distance() {
        let stmt = "VECTOR INDEX idx_vec (embedding) with(distance=cosine, type=hnsw, lib=vsag)";
        assert!(matches!(parse_distance(stmt), Some(DistanceMetric::Cosine)));
        let stmt2 = "with(distance=inner_product)";
        assert!(matches!(
            parse_distance(stmt2),
            Some(DistanceMetric::InnerProduct)
        ));
        assert_eq!(parse_distance("none"), None);
    }

    #[test]
    fn test_build_create_table_sql() {
        let sql = build_create_table_sql("c$v1$foo", 384, DistanceMetric::Cosine);
        assert!(sql.contains("c$v1$foo"));
        assert!(sql.contains("vector(384)"));
        assert!(sql.contains("distance=cosine"));
        assert!(sql.contains("FULLTEXT INDEX"));
    }
}
