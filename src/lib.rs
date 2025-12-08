//! SeekDB Rust SDK (server mode) – skeleton implementation.

mod backend;

pub mod admin;
pub mod collection;
pub mod config;
pub mod embedding;
pub mod error;
pub mod filters;
pub mod meta;
pub mod server;
pub mod types;

pub use crate::admin::{AdminApi, AdminClient};
pub use crate::collection::Collection;
pub use crate::config::{DistanceMetric, HnswConfig, ServerConfig};
pub use crate::embedding::EmbeddingFunction;
pub use crate::error::SeekDbError;
pub use crate::filters::{DocFilter, Filter, SqlWhere};
pub use crate::meta::{CollectionFieldNames, CollectionNames};
pub use crate::server::ServerClient;
pub use crate::types::Database;
pub use crate::types::{
    Document, Documents, Embedding, Embeddings, GetResult, IncludeField, Metadata, QueryResult,
};

#[cfg(feature = "embedding")]
pub use crate::embedding::DefaultEmbedding;
