use crate::error::Result;

/// Minimal row abstraction used by higher-level collection/admin logic.
///
/// This trait is intentionally small and does not expose sqlx-specific types,
/// so that future embedded backends can provide their own row implementations.
pub trait BackendRow {
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
