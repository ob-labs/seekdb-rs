//! Query helper utilities for parameterized queries.
//!
//! Provides Value-based SQL escaping/substitution; embedded mode uses
//! `QueryParam`-based `substitute_sql_params` in `embedded.rs`. These helpers
//! are kept for potential server/JSON-based use.

use crate::error::Result;
use serde_json::Value;

/// Escapes SQL string values for safe string substitution.
#[allow(dead_code)]
pub fn escape_sql_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "''")
        .replace('\0', "\\0")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\x1a', "\\Z")
}

/// Converts a JSON value to a SQL string representation.
#[allow(dead_code)]
pub fn value_to_sql_string(value: &Value) -> String {
    match value {
        Value::String(s) => format!("'{}'", escape_sql_string(s)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                f.to_string()
            } else {
                format!("'{}'", escape_sql_string(&n.to_string()))
            }
        }
        Value::Bool(b) => {
            if *b {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        Value::Null => "NULL".to_string(),
        other => format!("'{}'", escape_sql_string(&other.to_string())),
    }
}

/// Builds a SQL query string by substituting `?` placeholders with escaped values.
#[allow(dead_code)]
pub fn build_sql_with_params(sql_template: &str, params: &[Value]) -> Result<String> {
    let mut sql = sql_template.to_string();
    let mut param_index = 0;

    // Simple replacement: find ? and replace with parameter value
    while let Some(pos) = sql[param_index..].find('?') {
        let actual_pos = param_index + pos;
        if param_index >= params.len() {
            return Err(crate::error::SeekDbError::InvalidInput(
                "Not enough parameters for SQL query".into(),
            ));
        }
        let param_value = value_to_sql_string(&params[param_index]);
        sql.replace_range(actual_pos..actual_pos + 1, &param_value);
        param_index += 1;
    }

    if param_index < params.len() {
        return Err(crate::error::SeekDbError::InvalidInput(
            "Too many parameters for SQL query".into(),
        ));
    }

    Ok(sql)
}
