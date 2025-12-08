use thiserror::Error;

/// Common result type used across the SDK.
pub type Result<T> = std::result::Result<T, SeekDbError>;

/// Unified error enum surfaced by all public APIs.
#[derive(Error, Debug)]
pub enum SeekDbError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("sql error: {0}")]
    Sql(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<sqlx::Error> for SeekDbError {
    fn from(value: sqlx::Error) -> Self {
        match value {
            sqlx::Error::RowNotFound => SeekDbError::NotFound("row not found".into()),
            _ => SeekDbError::Sql(value.to_string()),
        }
    }
}
