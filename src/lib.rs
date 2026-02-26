//! SeekDB Rust SDK – skeleton implementation.

mod backend;
#[cfg(any(feature = "server", feature = "embedded"))]
mod client_trait;
#[cfg(feature = "server")]
mod sqlx_helper;
mod query_helper;

pub mod admin;
pub mod collection;
pub mod config;
pub mod embedding;
pub mod error;
pub mod filters;
pub mod meta;
pub mod server;
#[cfg(feature = "sync")]
pub mod sync;
pub mod types;

#[cfg(feature = "embedded")]
mod sys;

#[cfg(feature = "embedded")]
pub mod embedded;

pub use crate::admin::{AdminApi, AdminClient, AdminClientBuilder, ADMIN_BOOTSTRAP_DATABASE};
pub use crate::collection::{
    AddBatch, Collection, DeleteQuery, GetQuery, HybridKnn, HybridQuery, HybridRank,
    UpdateBatch, UpsertBatch,
};
pub use crate::backend::{row_to_json_values, BackendRow, QueryParam};
pub use crate::config::{
    DistanceMetric, EmbeddedConfig, FulltextIndexConfig, HnswConfig, ServerConfig,
    DEFAULT_DISTANCE_METRIC_STR, DEFAULT_VECTOR_DIMENSION,
};
pub use crate::embedding::EmbeddingFunction;
pub use crate::error::SeekDbError;
pub use crate::filters::{DocFilter, Filter, SqlWhere};
pub use crate::meta::{CollectionFieldNames, CollectionNames};
pub use crate::server::ServerClient;

#[cfg(any(feature = "server", feature = "embedded"))]
pub mod client;

#[cfg(any(feature = "server", feature = "embedded"))]
pub use crate::client::{Client, ClientBuilder};

#[cfg(any(feature = "server", feature = "embedded"))]
pub use crate::client_trait::SeekDbClient;
pub use crate::types::Database;
pub use crate::types::{
    Document, Documents, Embedding, Embeddings, GetResult, IncludeField, Metadata, QueryResult,
};

#[cfg(feature = "embedding")]
pub use crate::embedding::DefaultEmbedding;

#[cfg(feature = "sync")]
pub use crate::sync::{SyncCollection, SyncServerClient};

#[cfg(all(feature = "sync", feature = "embedded"))]
pub use crate::sync::{
    SyncEmbeddedClient, SyncEmbeddedClientBuilder, SyncEmbeddedCollection,
};

#[cfg(feature = "embedded")]
pub use crate::embedded::{EmbeddedClient, EmbeddedClientBuilder, EmbeddedDatabase};
