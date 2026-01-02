/// Helpers for naming collections and columns, mirroring Python `meta_info.py`.
use crate::error::{Result, SeekDbError};

pub struct CollectionNames;

impl CollectionNames {
    /// Physical table name prefix for collections.
    pub const TABLE_PREFIX: &'static str = "c$v1$";

    /// Maximum allowed physical table name length.
    ///
    /// This limit applies to the full table name, including `TABLE_PREFIX`.
    pub const MAX_TABLE_NAME_LEN: usize = 64;

    /// Validate a logical collection name.
    ///
    /// Current rules:
    /// - must be non-empty
    /// - must only contain ASCII letters, digits, or underscore: `[a-zA-Z0-9_]`
    /// - the resulting physical table name (`TABLE_PREFIX` + name) must not exceed `MAX_TABLE_NAME_LEN`
    pub fn validate(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(SeekDbError::InvalidInput(
                "collection name must not be empty".into(),
            ));
        }

        let physical_len = Self::TABLE_PREFIX.len() + name.len();
        if physical_len > Self::MAX_TABLE_NAME_LEN {
            return Err(SeekDbError::InvalidInput(format!(
                "collection name too long; physical table name exceeds {} characters",
                Self::MAX_TABLE_NAME_LEN
            )));
        }

        if !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(SeekDbError::InvalidInput(
                "collection name must match [a-zA-Z0-9_]".into(),
            ));
        }

        Ok(())
    }

    /// Build the physical table name for a collection.
    pub fn table_name(name: &str) -> String {
        // Keep identical naming to the Python client: c$v1${collection_name}
        format!("{}{}", Self::TABLE_PREFIX, name)
    }
}

/// Column name helpers.
pub struct CollectionFieldNames;

impl CollectionFieldNames {
    pub const ID: &'static str = "_id";
    pub const DOCUMENT: &'static str = "document";
    pub const EMBEDDING: &'static str = "embedding";
    pub const METADATA: &'static str = "metadata";
}

#[cfg(test)]
mod tests {
    use super::CollectionNames;
    use crate::error::SeekDbError;

    #[test]
    fn valid_collection_name_passes() {
        assert!(CollectionNames::validate("coll_123").is_ok());
    }

    #[test]
    fn empty_collection_name_fails() {
        let err = CollectionNames::validate("").unwrap_err();
        assert!(matches!(err, SeekDbError::InvalidInput(_)));
    }

    #[test]
    fn invalid_chars_collection_name_fails() {
        let err = CollectionNames::validate("bad-name").unwrap_err();
        assert!(matches!(err, SeekDbError::InvalidInput(_)));
    }

    #[test]
    fn too_long_collection_name_fails() {
        let allowed_len =
            CollectionNames::MAX_TABLE_NAME_LEN - CollectionNames::TABLE_PREFIX.len();
        let long_name = "a".repeat(allowed_len + 1);
        let err = CollectionNames::validate(&long_name).unwrap_err();
        assert!(matches!(err, SeekDbError::InvalidInput(_)));
    }
}
