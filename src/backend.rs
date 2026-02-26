use async_trait::async_trait;

use crate::error::Result;

/// Convert a BackendRow to a vec of `serde_json::Value` by column index (0..max_cols).
/// Uses `get_string_by_index` for each column; useful for callers that need JSON values for parsing.
pub fn row_to_json_values(row: &dyn BackendRow, max_cols: usize) -> Vec<serde_json::Value> {
    let mut v = Vec::with_capacity(max_cols);
    for i in 0..max_cols {
        match row.get_string_by_index(i) {
            Ok(Some(s)) => v.push(serde_json::Value::String(s)),
            Ok(None) => v.push(serde_json::Value::Null),
            Err(_) => break,
        }
    }
    v
}

/// Parameter for parameterized SQL (used by Collection with both server and embedded backends).
#[derive(Clone, Debug)]
pub enum QueryParam {
    String(String),
    Bytes(Vec<u8>),
    I64(i64),
    F32(f32),
    Null,
}

impl QueryParam {
    /// Build a query param from a metadata filter value (for WHERE clauses).
    pub fn from_metadata_value(v: &serde_json::Value) -> Self {
        use serde_json::Value as JsonValue;
        match v {
            JsonValue::String(s) => QueryParam::String(s.clone()),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    QueryParam::I64(i)
                } else if let Some(u) = n.as_u64() {
                    QueryParam::I64(u as i64)
                } else if let Some(f) = n.as_f64() {
                    QueryParam::F32(f as f32)
                } else {
                    QueryParam::String(n.to_string())
                }
            }
            JsonValue::Bool(b) => QueryParam::String(if *b { "1" } else { "0" }.to_string()),
            JsonValue::Null => QueryParam::Null,
            other => QueryParam::String(other.to_string()),
        }
    }
}

/// Backend used by Collection for SQL execution. Abstracts over server (sqlx pool)
/// and embedded (C API) so Collection can work with both.
#[async_trait]
pub trait CollectionBackend: Send + Sync {
    /// Execute a SQL statement that does not return rows.
    async fn execute(&self, sql: &str) -> Result<()>;

    /// Fetch all rows for the given SQL query (no parameters).
    async fn fetch_all(&self, sql: &str) -> Result<Vec<Box<dyn BackendRow>>>;

    /// Execute a SQL statement with ? placeholders; params are bound in order.
    async fn execute_with_params(&self, sql: &str, params: &[QueryParam]) -> Result<()>;

    /// Fetch all rows for the given SQL with ? placeholders.
    async fn fetch_all_with_params(&self, sql: &str, params: &[QueryParam]) -> Result<Vec<Box<dyn BackendRow>>>;
}

/// Minimal row abstraction used by higher-level collection/admin logic.
///
/// This trait is intentionally small and does not expose sqlx-specific types,
/// so that future embedded backends can provide their own row implementations.
/// Requires Send + Sync so that futures returning Vec<Box<dyn BackendRow>> are Send.
pub trait BackendRow: Send + Sync {
    /// Get a binary value from a column (commonly used for `_id`).
    fn get_bytes(&self, column: &str) -> Result<Option<Vec<u8>>>;

    /// Get a string value from a column (used for documents, JSON, etc.).
    fn get_string(&self, column: &str) -> Result<Option<String>>;

    /// Get a 32-bit float value from a column (used for distances/scores).
    fn get_f32(&self, column: &str) -> Result<Option<f32>>;

    /// Get a 64-bit integer value from a column (used for counts).
    fn get_i64(&self, column: &str) -> Result<Option<i64>>;

    /// Get a string value by column index (used for engine-generated aliases).
    fn get_string_by_index(&self, index: usize) -> Result<Option<String>>;
}

/// Asynchronous SQL backend abstraction.
///
/// This trait is defined for future embedded/server backends; for now it is
/// implemented only for `ServerClient`. Collection/admin code can gradually
/// migrate to depend on this trait instead of a concrete client.
#[async_trait::async_trait]
pub trait SqlBackend: Send + Sync {
    type Row: BackendRow + Send + Sync;

    /// Execute a SQL statement that does not return rows.
    async fn execute(&self, sql: &str) -> Result<()>;

    /// Fetch all rows for the given SQL query.
    async fn fetch_all(&self, sql: &str) -> Result<Vec<Self::Row>>;

    /// Return a short mode string (e.g., "server", "embedded") for logging.
    fn mode(&self) -> &'static str;
}

impl BackendRow for sqlx::mysql::MySqlRow {
    fn get_bytes(&self, column: &str) -> Result<Option<Vec<u8>>> {
        use sqlx::Row;
        let v = self.try_get::<Option<Vec<u8>>, _>(column);
        v.map_err(Into::into)
    }

    fn get_string(&self, column: &str) -> Result<Option<String>> {
        use sqlx::Row;
        let v = self.try_get::<Option<String>, _>(column);
        v.map_err(Into::into)
    }

    fn get_f32(&self, column: &str) -> Result<Option<f32>> {
        use sqlx::Row;
        // COUNT/score-style columns are non-null in normal queries; wrap into Option here.
        let v: std::result::Result<f32, sqlx::Error> = self.try_get(column);
        v.map(Some).map_err(Into::into)
    }

    fn get_i64(&self, column: &str) -> Result<Option<i64>> {
        use sqlx::Row;
        let v: std::result::Result<i64, sqlx::Error> = self.try_get(column);
        v.map(Some).map_err(Into::into)
    }

    fn get_string_by_index(&self, index: usize) -> Result<Option<String>> {
        use sqlx::Row;
        let v: std::result::Result<String, sqlx::Error> = self.try_get(index);
        v.map(Some).map_err(Into::into)
    }
}
