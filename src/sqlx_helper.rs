//! Helper utilities for sqlx-based queries
//!
//! This module provides utilities to work with sqlx in a way that can be
//! abstracted for both server and embedded modes.

#[cfg(feature = "server")]
use sqlx::mysql::MySqlPool;

/// Trait for clients that support sqlx-based parameterized queries.
///
/// This trait is implemented by ServerClient to provide access to the sqlx pool.
/// EmbeddedClient does not implement this trait, as it uses a different query mechanism.
#[cfg(feature = "server")]
#[allow(dead_code)]
pub trait SqlxClient {
    /// Get the underlying sqlx MySqlPool.
    fn pool(&self) -> &MySqlPool;
}

#[cfg(feature = "server")]
impl SqlxClient for crate::server::ServerClient {
    fn pool(&self) -> &MySqlPool {
        crate::server::ServerClient::pool(self)
    }
}
