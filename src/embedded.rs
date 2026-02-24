//! SeekDB Embedded Mode Client
//!
//! This module provides an embedded client for SeekDB that uses the native C library
//! directly, similar to Python's embedded client.

use async_trait::async_trait;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use crate::admin::AdminApi;
use crate::backend::{BackendRow, CollectionBackend, QueryParam, SqlBackend};
use crate::client_trait::SeekDbClient;
use crate::config::DistanceMetric;
use crate::collection::Collection;
use crate::config::{EmbeddedConfig, HnswConfig};
use crate::embedding::EmbeddingFunction;
use crate::error::{Result, SeekDbError};
use crate::meta::CollectionNames;
use crate::server::build_create_table_sql;
use crate::sys::*;
use crate::types::Database;

/// Builder for configuring and constructing an [`EmbeddedClient`].
pub struct EmbeddedClientBuilder {
    db_dir: String,
    database: String,
    autocommit: bool,
    port: Option<i32>,
    skip_open: bool,
}

/// Embedded client that uses the native SeekDB C library.
#[derive(Clone)]
pub struct EmbeddedClient {
    handle: Arc<EmbeddedHandle>,
    database: String,
}

// Internal handle wrapper that manages the C connection
struct EmbeddedHandle {
    handle: SeekdbHandle,
}

unsafe impl Send for EmbeddedHandle {}
unsafe impl Sync for EmbeddedHandle {}

impl Drop for EmbeddedHandle {
    fn drop(&mut self) {
        unsafe {
            seekdb_connect_close(self.handle);
        }
    }
}

// Embedded row implementation for BackendRow trait
// Note: We store actual data instead of raw pointers to avoid lifetime issues
pub struct EmbeddedRow {
    data: Vec<Option<String>>,  // Column data, indexed by column index
    column_names: Vec<String>,
    column_count: u32,
}

impl BackendRow for EmbeddedRow {
    fn get_bytes(&self, column: &str) -> Result<Option<Vec<u8>>> {
        let col_idx = self
            .column_names
            .iter()
            .position(|n| n == column)
            .ok_or_else(|| SeekDbError::InvalidInput(format!("column not found: {column}")))?;

        if col_idx >= self.data.len() {
            return Ok(None);
        }

        match &self.data[col_idx] {
            Some(s) => Ok(Some(s.as_bytes().to_vec())),
            None => Ok(None),
        }
    }

    fn get_string(&self, column: &str) -> Result<Option<String>> {
        let col_idx = self
            .column_names
            .iter()
            .position(|n| n == column)
            .ok_or_else(|| SeekDbError::InvalidInput(format!("column not found: {column}")))?;

        if col_idx >= self.data.len() {
            return Ok(None);
        }

        Ok(self.data[col_idx].clone())
    }

    fn get_f32(&self, column: &str) -> Result<Option<f32>> {
        let col_idx = self
            .column_names
            .iter()
            .position(|n| n == column)
            .ok_or_else(|| SeekDbError::InvalidInput(format!("column not found: {column}")))?;

        if col_idx >= self.data.len() {
            return Ok(None);
        }

        match &self.data[col_idx] {
            Some(s) => s.parse::<f32>().map(Some).map_err(|e| {
                SeekDbError::InvalidInput(format!("Failed to parse as f32: {}: {}", s, e))
            }),
            None => Ok(None),
        }
    }

    fn get_i64(&self, column: &str) -> Result<Option<i64>> {
        let col_idx = self
            .column_names
            .iter()
            .position(|n| n == column)
            .ok_or_else(|| SeekDbError::InvalidInput(format!("column not found: {column}")))?;

        if col_idx >= self.data.len() {
            return Ok(None);
        }

        match &self.data[col_idx] {
            Some(s) => s.parse::<i64>().map(Some).map_err(|e| {
                SeekDbError::InvalidInput(format!("Failed to parse as i64: {}: {}", s, e))
            }),
            None => Ok(None),
        }
    }

    fn get_string_by_index(&self, index: usize) -> Result<Option<String>> {
        if index >= self.column_count as usize {
            return Err(SeekDbError::InvalidInput(format!(
                "column index out of range: {index}"
            )));
        }

        if index >= self.data.len() {
            return Ok(None);
        }

        Ok(self.data[index].clone())
    }
}

impl EmbeddedClient {
    /// Build a client from an `EmbeddedConfig`.
    /// Runs seekdb_open on a blocking thread, then connect.
    /// If the database does not exist (C ABI returns "database is null"), creates it via AdminApi then reconnects (aligned with pyseekdb).
    pub async fn from_config(config: EmbeddedConfig) -> Result<Self> {
        let db_dir = config.db_dir.clone();
        let port = config.port;
        tokio::task::spawn_blocking(move || {
            if let Some(p) = port {
                EmbeddedDatabase::open_with_service(&db_dir, p)
            } else {
                EmbeddedDatabase::open(&db_dir)
            }
        })
        .await
        .map_err(|e| SeekDbError::Connection(format!("spawn_blocking open failed: {e}")))??;
        Self::connect_or_create_then_connect(
            &config.db_dir,
            &config.database,
            config.autocommit,
            config.port,
        )
        .await
    }

    /// Build a client from environment variables.
    pub async fn from_env() -> Result<Self> {
        let config = EmbeddedConfig::from_env()?;
        Self::from_config(config).await
    }

    pub fn database(&self) -> &str {
        &self.database
    }

    pub fn builder() -> EmbeddedClientBuilder {
        EmbeddedClientBuilder::new()
    }

    /// Execute a SQL statement that does not return rows.
    pub async fn execute(&self, sql: &str) -> Result<()> {
        let sql_cstr = CString::new(sql)?;
        let mut result: SeekdbResult = ptr::null_mut();

        let status = unsafe { seekdb_query(self.handle.handle, sql_cstr.as_ptr(), &mut result) };

        if status != SEEKDB_SUCCESS {
            let error_msg = unsafe {
                let err = seekdb_error(self.handle.handle);
                if err.is_null() {
                    format!("Query failed: error code {}", status)
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                }
            };
            return Err(SeekDbError::Sql(error_msg));
        }

        // For statements that return rows (SELECT), consume the result set by fetching until null.
        // Rows are valid until next seekdb_fetch_row() or seekdb_result_free(); no seekdb_row_free.
        if !result.is_null() {
            while !unsafe { seekdb_fetch_row(result) }.is_null() {}
            unsafe { seekdb_result_free(result); }
        }

        Ok(())
    }

    /// Fetch all rows for the given SQL query.
    pub async fn fetch_all(&self, sql: &str) -> Result<Vec<EmbeddedRow>> {
        let sql_cstr = CString::new(sql)?;
        let mut result: SeekdbResult = ptr::null_mut();

        let status = unsafe { seekdb_query(self.handle.handle, sql_cstr.as_ptr(), &mut result) };

        if status != SEEKDB_SUCCESS {
            let error_msg = unsafe {
                let err = seekdb_error(self.handle.handle);
                if err.is_null() {
                    format!("Query failed: error code {}", status)
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                }
            };
            return Err(SeekDbError::Sql(error_msg));
        }

        // Check if result is null (for statements that don't return rows)
        if result.is_null() {
            return Ok(Vec::new());
        }

        // Get column names using the new convenience API
        let num_cols = unsafe { seekdb_num_fields(result) };
        if num_cols == 0 || num_cols == (0u32).wrapping_sub(1) {
            // No columns, return empty result
            unsafe {
                seekdb_result_free(result);
            }
            return Ok(Vec::new());
        }
        
        let mut column_names = Vec::with_capacity(num_cols as usize);
        
        // Try using the new alloc API first, fallback to manual method
        let mut names_ptr: *mut *mut c_char = ptr::null_mut();
        let mut col_count: i32 = num_cols as i32;
        let status = unsafe {
            seekdb_result_get_all_column_names_alloc(result, &mut names_ptr, &mut col_count)
        };
        
        if status == SEEKDB_SUCCESS && !names_ptr.is_null() {
            // Successfully got column names from alloc API
            unsafe {
                for i in 0..col_count {
                    let name_ptr = *names_ptr.add(i as usize);
                    if !name_ptr.is_null() {
                        if let Ok(name) = CStr::from_ptr(name_ptr).to_str() {
                            column_names.push(name.to_string());
                        }
                    }
                }
                // Free the allocated memory
                seekdb_free_column_names(names_ptr, col_count);
            }
        } else {
            // Fallback to manual method
            for i in 0..num_cols {
                let mut name_buf = vec![0u8; 256];
                let status = unsafe {
                    seekdb_result_column_name(
                        result,
                        i as i32,
                        name_buf.as_mut_ptr() as *mut c_char,
                        name_buf.len(),
                    )
                };
                if status == SEEKDB_SUCCESS {
                    if let Some(null_pos) = name_buf.iter().position(|&b| b == 0) {
                        name_buf.truncate(null_pos);
                    }
                    if let Ok(name) = String::from_utf8(name_buf) {
                        column_names.push(name);
                    }
                }
            }
        }

        // Fetch all rows and copy data immediately to avoid lifetime issues
        // According to C++ implementation, seekdb_fetch_row returns a SeekdbRowData*
        // that holds a reference to the result set. We need to copy data immediately
        // and free each row after copying.
        let num_rows = unsafe { seekdb_num_rows(result) };
        let mut rows = Vec::with_capacity(num_rows as usize);

        loop {
            let row = unsafe { seekdb_fetch_row(result) };
            if row.is_null() {
                break;
            }

            // Copy all column data immediately
            let mut row_data = Vec::with_capacity(num_cols as usize);
            for col_idx in 0..num_cols {
                let is_null = unsafe { seekdb_row_is_null(row, col_idx as i32) };
                if is_null {
                    row_data.push(None);
                } else {
                    // Try to get as string first (most common case)
                    let len = unsafe { seekdb_row_get_string_len(row, col_idx as i32) };
                    if len != (0usize).wrapping_sub(1) {
                        let mut buf = vec![0u8; len + 1];
                        let status = unsafe {
                            seekdb_row_get_string(
                                row,
                                col_idx as i32,
                                buf.as_mut_ptr() as *mut c_char,
                                buf.len(),
                            )
                        };
                        if status == SEEKDB_SUCCESS {
                            if buf.last() == Some(&0) {
                                buf.pop();
                            }
                            if let Ok(s) = String::from_utf8(buf) {
                                row_data.push(Some(s));
                            } else {
                                row_data.push(None);
                            }
                        } else {
                            // Try as integer
                            let mut int_val: i64 = 0;
                            if unsafe { seekdb_row_get_int64(row, col_idx as i32, &mut int_val) } == SEEKDB_SUCCESS {
                                row_data.push(Some(int_val.to_string()));
                            } else {
                                // Try as double
                                let mut double_val: f64 = 0.0;
                                if unsafe { seekdb_row_get_double(row, col_idx as i32, &mut double_val) } == SEEKDB_SUCCESS {
                                    row_data.push(Some(double_val.to_string()));
                                } else {
                                    row_data.push(None);
                                }
                            }
                        }
                    } else {
                        // Try as integer
                        let mut int_val: i64 = 0;
                        if unsafe { seekdb_row_get_int64(row, col_idx as i32, &mut int_val) } == SEEKDB_SUCCESS {
                            row_data.push(Some(int_val.to_string()));
                        } else {
                            // Try as double
                            let mut double_val: f64 = 0.0;
                            if unsafe { seekdb_row_get_double(row, col_idx as i32, &mut double_val) } == SEEKDB_SUCCESS {
                                row_data.push(Some(double_val.to_string()));
                            } else {
                                row_data.push(None);
                            }
                        }
                    }
                }
            }

            rows.push(EmbeddedRow {
                data: row_data,
                column_names: column_names.clone(),
                column_count: num_cols,
            });

            // Note: According to C++ implementation, seekdb_fetch_row() automatically
            // frees the previous row_data when fetching the next row, so we don't need
            // to explicitly free each row during iteration. The last row will be freed
            // when we free the result.
        }

        // Free result now that we've copied all data and freed all rows
        unsafe {
            seekdb_result_free(result);
        }

        Ok(rows)
    }

    pub async fn create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        CollectionNames::validate(name)?;

        let cfg = config.ok_or_else(|| {
            SeekDbError::Config("HnswConfig must be provided when creating a collection".into())
        })?;

        let table_name = CollectionNames::table_name(name);
        let sql = build_create_table_sql(&table_name, cfg.dimension, cfg.distance);
        self.execute(&sql).await?;

        Ok(Collection::new(
            Arc::new(self.clone()),
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
        CollectionNames::validate(name)?;
        let table_name = CollectionNames::table_name(name);

        let describe_sql = format!("DESCRIBE `{table_name}`");
        let describe = self.fetch_all(&describe_sql).await?;
        if describe.is_empty() {
            return Err(SeekDbError::NotFound(format!(
                "collection not found: {name}"
            )));
        }

        let mut dimension: Option<u32> = None;
        for row in &describe {
            let field = row.get_string("Field").ok().flatten().unwrap_or_default();
            if field == "embedding" {
                let type_str = row.get_string("Type").ok().flatten().unwrap_or_default();
                dimension = parse_dimension_from_type(&type_str);
                break;
            }
        }

        let create_sql = format!("SHOW CREATE TABLE `{table_name}`");
        let create_rows = self.fetch_all(&create_sql).await?;
        let mut distance = DistanceMetric::L2;
        if let Some(row) = create_rows.first() {
            let create_stmt = row
                .get_string("Create Table")
                .ok()
                .flatten()
                .or_else(|| row.get_string_by_index(1).ok().flatten())
                .unwrap_or_default();
            if let Some(d) = parse_distance_from_create(&create_stmt) {
                distance = d;
            }
        }

        let dimension = dimension.ok_or_else(|| {
            SeekDbError::Config("cannot detect dimension from collection schema".into())
        })?;

        Ok(Collection::new(
            Arc::new(self.clone()),
            name.to_string(),
            None,
            dimension,
            distance,
            embedding_function,
            None,
        ))
    }

    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        CollectionNames::validate(name)?;
        let table_name = CollectionNames::table_name(name);
        let sql = format!("DROP TABLE IF EXISTS `{table_name}`");
        self.execute(&sql).await
    }

    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let prefix = CollectionNames::TABLE_PREFIX;
        let like_pattern = format!("{prefix}%");
        let show_sql = format!("SHOW TABLES LIKE '{like_pattern}'");

        let rows = self.fetch_all(&show_sql).await?;

        let mut names = Vec::new();
        for row in rows {
            if let Ok(Some(table_name)) = row.get_string_by_index(0) {
                if let Some(name) = table_name.strip_prefix(CollectionNames::TABLE_PREFIX) {
                    names.push(name.to_string());
                }
            }
        }
        Ok(names)
    }

    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        CollectionNames::validate(name)?;
        let table_name = CollectionNames::table_name(name);
        
        // For embedded mode, we need to use a different approach
        // Since we don't have parameterized queries yet, use string formatting
        let sql = format!(
            "SELECT 1 FROM information_schema.TABLES \
             WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME = '{}' LIMIT 1",
            self.database,
            table_name.replace('\'', "''")
        );
        
        let rows = self.fetch_all(&sql).await?;
        Ok(!rows.is_empty())
    }

    pub async fn get_or_create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        CollectionNames::validate(name)?;

        if self.has_collection(name).await? {
            self.get_collection(name, embedding_function).await
        } else {
            self.create_collection(name, config, embedding_function).await
        }
    }

    pub async fn count_collection(&self) -> Result<usize> {
        let collections = self.list_collections().await?;
        Ok(collections.len())
    }

    async fn connect_internal(
        db_dir: &str,
        database: &str,
        autocommit: bool,
        port: Option<i32>,
    ) -> Result<Self> {
        // Caller must have opened via EmbeddedDatabase::open (in from_config/build we use spawn_blocking).
        let _ = (db_dir, port);
        let db_name_cstr = CString::new(database)?;
        let mut handle: SeekdbHandle = ptr::null_mut();

        let result = unsafe { seekdb_connect(&mut handle, db_name_cstr.as_ptr(), autocommit) };

        if result != SEEKDB_SUCCESS {
            let error_msg = unsafe {
                let err = seekdb_last_error();
                if err.is_null() {
                    format!("Failed to connect: error code {}", result)
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                }
            };
            return Err(SeekDbError::Connection(error_msg));
        }

        if handle.is_null() {
            return Err(SeekDbError::Connection("seekdb_connect returned null handle".into()));
        }

        Ok(Self {
            handle: Arc::new(EmbeddedHandle { handle }),
            database: database.to_string(),
        })
    }

    /// Connect to the given database; if C ABI returns "database is null" (DB does not exist),
    /// connect to information_schema, create the DB (CREATE DATABASE IF NOT EXISTS), then reconnect.
    /// Aligned with pyseekdb: use system DB (information_schema) for bootstrap, no empty database.
    async fn connect_or_create_then_connect(
        db_dir: &str,
        database: &str,
        autocommit: bool,
        port: Option<i32>,
    ) -> Result<Self> {
        match Self::connect_internal(db_dir, database, autocommit, port).await {
            Err(e) if e.to_string().contains("database is null") => {
                let bootstrap = Self::connect_internal(
                    db_dir,
                    crate::admin::ADMIN_BOOTSTRAP_DATABASE,
                    autocommit,
                    port,
                )
                .await?;
                let escaped = database.replace('`', "``");
                let sql = format!("CREATE DATABASE IF NOT EXISTS `{escaped}`");
                bootstrap.execute(&sql).await?;
                Self::connect_internal(db_dir, database, autocommit, port).await
            }
            other => other,
        }
    }
}

#[async_trait]
impl SqlBackend for EmbeddedClient {
    type Row = EmbeddedRow;

    async fn execute(&self, sql: &str) -> Result<()> {
        EmbeddedClient::execute(self, sql).await
    }

    async fn fetch_all(&self, sql: &str) -> Result<Vec<Self::Row>> {
        EmbeddedClient::fetch_all(self, sql).await
    }

    fn mode(&self) -> &'static str {
        "embedded"
    }
}

#[async_trait]
impl CollectionBackend for EmbeddedClient {
    async fn execute(&self, sql: &str) -> Result<()> {
        EmbeddedClient::execute(self, sql).await
    }

    async fn fetch_all(&self, sql: &str) -> Result<Vec<Box<dyn BackendRow>>> {
        let rows = EmbeddedClient::fetch_all(self, sql).await?;
        Ok(rows
            .into_iter()
            .map(|r| Box::new(r) as Box<dyn BackendRow>)
            .collect())
    }

    async fn execute_with_params(&self, sql: &str, params: &[QueryParam]) -> Result<()> {
        let sql = substitute_sql_params(sql, params)?;
        self.execute(&sql).await
    }

    async fn fetch_all_with_params(&self, sql: &str, params: &[QueryParam]) -> Result<Vec<Box<dyn BackendRow>>> {
        let sql = substitute_sql_params(sql, params)?;
        let rows = EmbeddedClient::fetch_all(self, &sql).await?;
        Ok(rows
            .into_iter()
            .map(|r| Box::new(r) as Box<dyn BackendRow>)
            .collect())
    }
}

/// Substitute `?` placeholders with escaped parameter values.
/// Returns an error if the number of `?` in `sql` does not match `params.len()`.
fn substitute_sql_params(sql: &str, params: &[QueryParam]) -> Result<String> {
    let placeholders = sql.matches('?').count();
    if placeholders != params.len() {
        return Err(SeekDbError::InvalidInput(format!(
            "SQL has {} placeholders but {} params provided",
            placeholders,
            params.len()
        )));
    }
    let mut out = String::with_capacity(sql.len() + params.iter().fold(0, |acc, p| acc + estimate_param_len(p)));
    let mut param_iter = params.iter();
    for part in sql.split('?') {
        out.push_str(part);
        if let Some(p) = param_iter.next() {
            let s = match p {
                QueryParam::String(s) => {
                    let escaped = s.replace('\\', "\\\\").replace('\'', "''");
                    format!("'{escaped}'")
                }
                QueryParam::Bytes(b) => {
                    let hex: String = b.iter().map(|x| format!("{:02x}", x)).collect();
                    format!("X'{hex}'")
                }
                QueryParam::I64(i) => i.to_string(),
                QueryParam::F32(f) => f.to_string(),
                QueryParam::Null => "NULL".to_string(),
            };
            out.push_str(&s);
        }
    }
    Ok(out)
}

#[inline]
fn estimate_param_len(p: &QueryParam) -> usize {
    match p {
        QueryParam::String(s) => s.len() + 4,
        QueryParam::Bytes(b) => b.len() * 2 + 4,
        QueryParam::I64(_) => 24,
        QueryParam::F32(_) => 24,
        QueryParam::Null => 4,
    }
}

fn parse_dimension_from_type(type_str: &str) -> Option<u32> {
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

fn parse_distance_from_create(create_stmt: &str) -> Option<DistanceMetric> {
    let lower = create_stmt.to_lowercase();
    if let Some(pos) = lower.find("distance=") {
        let rest = &lower[pos + "distance=".len()..];
        let value: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        return match value.as_str() {
            "l2" => Some(DistanceMetric::L2),
            "cosine" => Some(DistanceMetric::Cosine),
            "inner_product" | "ip" => Some(DistanceMetric::InnerProduct),
            _ => None,
        };
    }
    None
}

// Implement SeekDbClient trait for EmbeddedClient
#[async_trait]
impl SeekDbClient for EmbeddedClient {
    fn database(&self) -> &str {
        &self.database
    }

    async fn create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        EmbeddedClient::create_collection(self, name, config, embedding_function).await
    }

    async fn get_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        EmbeddedClient::get_collection(self, name, embedding_function).await
    }

    async fn delete_collection(&self, name: &str) -> Result<()> {
        EmbeddedClient::delete_collection(self, name).await
    }

    async fn list_collections(&self) -> Result<Vec<String>> {
        EmbeddedClient::list_collections(self).await
    }

    async fn has_collection(&self, name: &str) -> Result<bool> {
        EmbeddedClient::has_collection(self, name).await
    }

    async fn get_or_create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>> {
        EmbeddedClient::get_or_create_collection(self, name, config, embedding_function).await
    }

    async fn count_collection(&self) -> Result<usize> {
        EmbeddedClient::count_collection(self).await
    }
}

#[async_trait]
impl AdminApi for EmbeddedClient {
    async fn create_database(&self, name: &str, _tenant: Option<&str>) -> Result<()> {
        let sql = format!("CREATE DATABASE IF NOT EXISTS `{}`", name.replace('`', "``"));
        self.execute(&sql).await
    }

    async fn get_database(&self, name: &str, _tenant: Option<&str>) -> Result<Database> {
        // For embedded mode, we'll query information_schema
        let sql = format!(
            "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME \
             FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '{}'",
            name.replace('\'', "''")
        );
        let rows = self.fetch_all(&sql).await?;

        if rows.is_empty() {
            return Err(SeekDbError::NotFound(format!("database not found: {name}")));
        }

        let row = &rows[0];
        Ok(Database {
            name: row
                .get_string("SCHEMA_NAME")?
                .unwrap_or_else(|| name.to_string()),
            tenant: None, // Embedded mode doesn't use tenants
            charset: row.get_string("DEFAULT_CHARACTER_SET_NAME")?,
            collation: row.get_string("DEFAULT_COLLATION_NAME")?,
        })
    }

    async fn delete_database(&self, name: &str, _tenant: Option<&str>) -> Result<()> {
        let sql = format!("DROP DATABASE IF EXISTS `{}`", name.replace('`', "``"));
        self.execute(&sql).await
    }

    async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        _tenant: Option<&str>,
    ) -> Result<Vec<Database>> {
        let mut sql = String::from(
            "SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME \
             FROM information_schema.SCHEMATA",
        );

        if let Some(lim) = limit {
            sql.push_str(&format!(" LIMIT {}", lim));
            if let Some(off) = offset {
                sql.push_str(&format!(" OFFSET {}", off));
            }
        }

        let rows = self.fetch_all(&sql).await?;
        let mut databases = Vec::new();

        for row in rows {
            databases.push(Database {
                name: row.get_string("SCHEMA_NAME")?.unwrap_or_default(),
                tenant: None,
                charset: row.get_string("DEFAULT_CHARACTER_SET_NAME")?,
                collation: row.get_string("DEFAULT_COLLATION_NAME")?,
            });
        }

        Ok(databases)
    }
}

impl EmbeddedClientBuilder {
    fn new() -> Self {
        Self {
            db_dir: "seekdb.db".to_string(),
            database: "test".to_string(),
            autocommit: false,
            port: None,
            skip_open: false,
        }
    }

    /// When true, build() only connects and does not call `EmbeddedDatabase::open`.
    /// Use when the database was already opened (e.g. by `run_embedded_tests` on the main thread).
    pub fn skip_open(mut self, skip: bool) -> Self {
        self.skip_open = skip;
        self
    }

    pub fn db_dir(mut self, db_dir: impl Into<String>) -> Self {
        self.db_dir = db_dir.into();
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    pub fn autocommit(mut self, autocommit: bool) -> Self {
        self.autocommit = autocommit;
        self
    }

    pub fn port(mut self, port: Option<i32>) -> Self {
        self.port = port;
        self
    }

    /// Load configuration from environment variables and apply to builder.
    pub fn from_env(mut self) -> Result<Self> {
        let config = EmbeddedConfig::from_env()?;
        self.db_dir = config.db_dir;
        self.database = config.database;
        self.autocommit = config.autocommit;
        self.port = config.port;
        Ok(self)
    }

    pub async fn build(self) -> Result<EmbeddedClient> {
        if self.db_dir.trim().is_empty() {
            return Err(SeekDbError::Config("db_dir must be non-empty".into()));
        }
        if self.database.trim().is_empty() {
            return Err(SeekDbError::Config("database must be non-empty".into()));
        }
        if !self.skip_open {
            let db_dir = self.db_dir.clone();
            let port = self.port;
            tokio::task::spawn_blocking(move || {
                if let Some(port) = port {
                    EmbeddedDatabase::open_with_service(&db_dir, port)
                } else {
                    EmbeddedDatabase::open(&db_dir)
                }
            })
            .await
            .map_err(|e| SeekDbError::Connection(format!("spawn_blocking open failed: {e}")))??;
        }
        EmbeddedClient::connect_or_create_then_connect(
            &self.db_dir,
            &self.database,
            self.autocommit,
            self.port,
        )
        .await
    }
}

/// Embedded database manager for opening/closing the database.
pub struct EmbeddedDatabase;

impl EmbeddedDatabase {
    /// Open an embedded database
    pub fn open<P: AsRef<Path>>(db_dir: P) -> Result<()> {
        let db_path = db_dir.as_ref().to_string_lossy();
        let db_cstr = CString::new(db_path.as_ref())?;

        let result = unsafe { seekdb_open(db_cstr.as_ptr()) };

        if result != SEEKDB_SUCCESS {
            let error_msg = unsafe {
                let err = seekdb_last_error();
                if err.is_null() {
                    format!("Failed to open database: error code {}", result)
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                }
            };
            return Err(SeekDbError::Connection(error_msg));
        }

        Ok(())
    }

    /// Open an embedded database with service (network) support
    pub fn open_with_service<P: AsRef<Path>>(db_dir: P, port: i32) -> Result<()> {
        let db_path = db_dir.as_ref().to_string_lossy();
        let db_cstr = CString::new(db_path.as_ref())?;

        let result = unsafe { seekdb_open_with_service(db_cstr.as_ptr(), port) };

        if result != SEEKDB_SUCCESS {
            let error_msg = unsafe {
                let err = seekdb_last_error();
                if err.is_null() {
                    format!("Failed to open database with service: error code {}", result)
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                }
            };
            return Err(SeekDbError::Connection(error_msg));
        }

        Ok(())
    }

    /// Close the embedded database (sync).
    /// Embedded mode: no-op. Same as seekdb-js internal-client-embedded.ts — we do not call
    /// seekdb_close() because: (1) DB is process-local, no need to manually close;
    /// (2) seekdb_close() may block (fsync, locks, background threads) and block the event loop.
    pub fn close() {
        // No-op for embedded: avoid C ABI seekdb_close() blocking/hang
    }

    /// Close the embedded database (async). No-op for same reason as close().
    pub async fn close_async() {
        // No-op for embedded: avoid calling seekdb_close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::QueryParam;

    #[test]
    fn test_substitute_sql_params_string() {
        let sql = "SELECT * FROM t WHERE name = ?";
        let params = [QueryParam::String("a'b".to_string())];
        let out = substitute_sql_params(sql, &params).unwrap();
        assert_eq!(out, "SELECT * FROM t WHERE name = 'a''b'");
    }

    #[test]
    fn test_substitute_sql_params_i64_f32_null() {
        let sql = "SELECT ? , ? , ?";
        let params = [
            QueryParam::I64(42),
            QueryParam::F32(1.5),
            QueryParam::Null,
        ];
        let out = substitute_sql_params(sql, &params).unwrap();
        assert_eq!(out, "SELECT 42 , 1.5 , NULL");
    }

    #[test]
    fn test_substitute_sql_params_bytes() {
        let sql = "SELECT ?";
        let params = [QueryParam::Bytes(vec![0xde, 0xad])];
        let out = substitute_sql_params(sql, &params).unwrap();
        assert_eq!(out, "SELECT X'dead'");
    }

    #[test]
    fn test_substitute_sql_params_placeholder_count_mismatch() {
        let sql = "SELECT ? , ?";
        let params = [QueryParam::I64(1)];
        let err = substitute_sql_params(sql, &params).unwrap_err();
        assert!(matches!(err, SeekDbError::InvalidInput(_)));
        assert!(err.to_string().contains("2 placeholders"));
        assert!(err.to_string().contains("1 params"));
    }

    #[test]
    fn test_substitute_sql_params_escape_backslash() {
        let sql = "SELECT ?";
        let params = [QueryParam::String("a\\b".to_string())];
        let out = substitute_sql_params(sql, &params).unwrap();
        assert_eq!(out, "SELECT 'a\\\\b'");
    }

    #[test]
    fn test_parse_dimension_from_type() {
        assert_eq!(parse_dimension_from_type("vector(384)"), Some(384));
        assert_eq!(parse_dimension_from_type("VECTOR( 10 )"), Some(10));
        assert_eq!(parse_dimension_from_type("int"), None);
        assert_eq!(parse_dimension_from_type("vector()"), None);
    }

    #[tokio::test]
    async fn test_builder_rejects_empty_db_dir() {
        let res = EmbeddedClient::builder()
            .db_dir("")
            .database("test")
            .build()
            .await;
        match &res {
            Err(e) => assert!(e.to_string().to_lowercase().contains("db_dir")),
            Ok(_) => panic!("expected Err for empty db_dir"),
        }
    }

    #[tokio::test]
    async fn test_builder_rejects_empty_database() {
        let res = EmbeddedClient::builder()
            .db_dir("/tmp/any")
            .database("")
            .build()
            .await;
        match &res {
            Err(e) => assert!(e.to_string().to_lowercase().contains("database")),
            Ok(_) => panic!("expected Err for empty database"),
        }
    }

    #[test]
    fn test_parse_distance_from_create() {
        assert_eq!(
            parse_distance_from_create("CREATE TABLE t ( distance=l2 )"),
            Some(DistanceMetric::L2)
        );
        assert_eq!(
            parse_distance_from_create("distance=cosine"),
            Some(DistanceMetric::Cosine)
        );
        assert_eq!(
            parse_distance_from_create("distance=inner_product"),
            Some(DistanceMetric::InnerProduct)
        );
        assert_eq!(
            parse_distance_from_create("distance=ip"),
            Some(DistanceMetric::InnerProduct)
        );
        assert_eq!(parse_distance_from_create("no distance here"), None);
    }
}
