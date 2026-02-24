use std::collections::HashMap;
use std::env;

use crate::error::{Result, SeekDbError};

/// Default vector dimension (align with pyseekdb / DefaultEmbeddingFunction 384).
pub const DEFAULT_VECTOR_DIMENSION: u32 = 384;
/// Default distance metric (align with pyseekdb).
pub const DEFAULT_DISTANCE_METRIC_STR: &str = "cosine";

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
            DistanceMetric::L2 => "l2",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::InnerProduct => "inner_product",
        }
    }

    /// Parse from string (e.g. "l2", "cosine", "inner_product").
    pub fn from_str(s: &str) -> Result<Self> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "l2" => Ok(DistanceMetric::L2),
            "cosine" => Ok(DistanceMetric::Cosine),
            "inner_product" | "ip" => Ok(DistanceMetric::InnerProduct),
            _ => Err(SeekDbError::Config(format!(
                "distance must be one of [l2, cosine, inner_product], got: {}",
                s
            ))),
        }
    }
}

/// Fulltext analyzer configuration (align with pyseekdb FulltextIndexConfig).
#[derive(Clone, Debug, Default)]
pub struct FulltextIndexConfig {
    /// Analyzer name: e.g. "ik", "space", "ngram", "ngram2", "beng".
    pub analyzer: String,
    /// Optional parser-specific parameters.
    pub properties: Option<HashMap<String, String>>,
}

impl FulltextIndexConfig {
    pub fn new(analyzer: impl Into<String>) -> Self {
        Self {
            analyzer: analyzer.into(),
            properties: None,
        }
    }

    pub fn with_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.properties = Some(properties);
        self
    }
}

/// HNSW configuration used during collection creation (align with pyseekdb HNSWConfiguration).
#[derive(Clone, Debug)]
pub struct HnswConfig {
    pub dimension: u32,
    pub distance: DistanceMetric,
}

impl HnswConfig {
    /// Create and validate: dimension must be positive, distance valid.
    pub fn new(dimension: u32, distance: DistanceMetric) -> Result<Self> {
        if dimension == 0 {
            return Err(SeekDbError::Config(
                "dimension must be positive, got 0".into(),
            ));
        }
        Ok(Self { dimension, distance })
    }

    /// Build with default distance (cosine).
    pub fn with_dimension(dimension: u32) -> Result<Self> {
        Self::new(dimension, DistanceMetric::Cosine)
    }
}

fn require_env(key: &str) -> Result<String> {
    env::var(key).map_err(|_| SeekDbError::Config(format!("missing env: {key}")))
}

/// Embedded database configuration for SeekDB embedded mode.
#[derive(Clone, Debug)]
pub struct EmbeddedConfig {
    pub db_dir: String,
    pub database: String,
    pub autocommit: bool,
    pub port: Option<i32>, // None for embedded mode, Some(port) for server mode
}

impl EmbeddedConfig {
    /// Build configuration from environment variables.
    ///
    /// Environment variable naming follows the same pattern as `ServerConfig`:
    /// - `EMBEDDED_DB_DIR` (required): database directory path
    /// - `EMBEDDED_DATABASE` (required): database name (matches `SERVER_DATABASE` naming)
    /// - `EMBEDDED_PORT` (optional): port number (matches `SERVER_PORT` naming, default: None for embedded mode)
    /// - `EMBEDDED_AUTOCOMMIT` (optional): autocommit mode (default: false)
    pub fn from_env() -> Result<Self> {
        let db_dir = require_env("EMBEDDED_DB_DIR")?;
        let database = require_env("EMBEDDED_DATABASE")?;
        let port = parse_env("EMBEDDED_PORT");
        let autocommit = parse_env("EMBEDDED_AUTOCOMMIT").unwrap_or(false);

        Ok(Self {
            db_dir,
            database,
            autocommit,
            port,
        })
    }
}

fn parse_env<T>(key: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    env::var(key).ok().and_then(|v| v.parse::<T>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hnsw_valid_configuration() {
        let c = HnswConfig::new(128, DistanceMetric::Cosine).unwrap();
        assert_eq!(c.dimension, 128);
        assert_eq!(c.distance.as_str(), "cosine");
    }

    #[test]
    fn hnsw_default_distance() {
        let c = HnswConfig::with_dimension(128).unwrap();
        assert_eq!(c.distance.as_str(), "cosine");
    }

    #[test]
    fn hnsw_invalid_dimension_zero() {
        let err = HnswConfig::new(0, DistanceMetric::L2).unwrap_err();
        assert!(matches!(err, SeekDbError::Config(_)));
        assert!(err.to_string().contains("dimension must be positive"));
    }

    #[test]
    fn hnsw_invalid_dimension_negative() {
        // dimension is u32 so -1 is not representable; we only need to test 0
        let err = HnswConfig::new(0, DistanceMetric::Cosine).unwrap_err();
        assert!(err.to_string().contains("positive"));
    }

    #[test]
    fn distance_from_str() {
        assert_eq!(DistanceMetric::from_str("l2").unwrap(), DistanceMetric::L2);
        assert_eq!(DistanceMetric::from_str("cosine").unwrap(), DistanceMetric::Cosine);
        assert_eq!(DistanceMetric::from_str("L2").unwrap(), DistanceMetric::L2);
        assert_eq!(DistanceMetric::from_str("inner_product").unwrap(), DistanceMetric::InnerProduct);
        assert_eq!(DistanceMetric::from_str("ip").unwrap(), DistanceMetric::InnerProduct);
    }

    #[test]
    fn distance_from_str_invalid() {
        let err = DistanceMetric::from_str("invalid").unwrap_err();
        assert!(matches!(err, SeekDbError::Config(_)));
        assert!(err.to_string().contains("l2"));
    }

    #[test]
    fn fulltext_index_config_default() {
        let c = FulltextIndexConfig::new("ik");
        assert_eq!(c.analyzer, "ik");
        assert!(c.properties.is_none());
    }

    #[test]
    fn fulltext_index_config_with_properties() {
        let mut props = HashMap::new();
        props.insert("size".to_string(), "2".to_string());
        let c = FulltextIndexConfig::new("ngram").with_properties(props.clone());
        assert_eq!(c.analyzer, "ngram");
        assert_eq!(c.properties.as_ref().unwrap().get("size"), Some(&"2".to_string()));
    }

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_VECTOR_DIMENSION, 384);
        assert_eq!(DEFAULT_DISTANCE_METRIC_STR, "cosine");
    }
}
