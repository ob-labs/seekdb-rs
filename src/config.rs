use std::env;

use crate::error::{Result, SeekDbError};

/// Server connection configuration for SeekDB over MySQL protocol.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub tenant: String,
    pub database: String,
    pub user: String,
    pub password: String,
    pub max_connections: u32,
}

impl ServerConfig {
    /// Build configuration from environment variables:
    /// `SERVER_HOST`, `SERVER_PORT`, `SERVER_TENANT`, `SERVER_DATABASE`,
    /// `SERVER_USER`, `SERVER_PASSWORD`, `SERVER_MAX_CONNECTIONS` (optional, default 5).
    pub fn from_env() -> Result<Self> {
        let host = require_env("SERVER_HOST")?;
        let port = parse_env("SERVER_PORT").unwrap_or(2881);
        let tenant = require_env("SERVER_TENANT")?;
        let database = require_env("SERVER_DATABASE")?;
        let user = require_env("SERVER_USER")?;
        let password = require_env("SERVER_PASSWORD")?;
        let max_connections = parse_env("SERVER_MAX_CONNECTIONS").unwrap_or(5);

        Ok(Self {
            host,
            port,
            tenant,
            database,
            user,
            password,
            max_connections,
        })
    }
}

/// Supported vector distance metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    L2,
    Cosine,
    InnerProduct,
}

impl DistanceMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetric::L2 => "L2",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::InnerProduct => "inner_product",
        }
    }
}

/// HNSW configuration used during collection creation.
#[derive(Clone, Debug)]
pub struct HnswConfig {
    pub dimension: u32,
    pub distance: DistanceMetric,
}

fn require_env(key: &str) -> Result<String> {
    env::var(key).map_err(|_| SeekDbError::Config(format!("missing env: {key}")))
}

fn parse_env<T>(key: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    env::var(key).ok().and_then(|v| v.parse::<T>().ok())
}
