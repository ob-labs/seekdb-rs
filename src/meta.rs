/// Helpers for naming collections and columns, mirroring Python `meta_info.py`.
use crate::error::{Result, SeekDbError};

pub struct CollectionNames;

impl CollectionNames {
    /// Physical table name prefix for collections (v1).
    pub const TABLE_PREFIX: &'static str = "c$v1$";
    /// Prefix for v2 collection tables (by collection id).
    pub const TABLE_PREFIX_V2: &'static str = "c$v2$";

    /// Maximum allowed length for user-facing collection names (align with pyseekdb).
    pub const MAX_COLLECTION_NAME_LENGTH: usize = 512;

    /// Validate a logical collection name.
    ///
    /// Rules (align with pyseekdb):
    /// - must be non-empty
    /// - length between 1 and MAX_COLLECTION_NAME_LENGTH (512)
    /// - only [a-zA-Z0-9_]
    pub fn validate(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(SeekDbError::InvalidInput(
                "collection name must not be empty".into(),
            ));
        }

        if name.len() > Self::MAX_COLLECTION_NAME_LENGTH {
            return Err(SeekDbError::InvalidInput(format!(
                "collection name too long: {} characters; maximum allowed is {}",
                name.len(),
                Self::MAX_COLLECTION_NAME_LENGTH
            )));
        }

        if !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(SeekDbError::InvalidInput(
                "collection name contains invalid characters. Only letters, digits, and underscore are allowed: [a-zA-Z0-9_]".into(),
            ));
        }

        Ok(())
    }

    /// Build the physical table name for a collection (v1).
    pub fn table_name(name: &str) -> String {
        format!("{}{}", Self::TABLE_PREFIX, name)
    }

    /// Build the physical table name for a collection by id (v2).
    pub fn table_name_v2(collection_id: &str) -> String {
        format!("{}{}", Self::TABLE_PREFIX_V2, collection_id)
    }

    /// Extract collection name from a table name (v1 prefix).
    pub fn collection_name(table_name: &str) -> &str {
        if let Some(name) = table_name.strip_prefix(Self::TABLE_PREFIX) {
            name
        } else {
            table_name
        }
    }

    /// Check if a table name is a collection table (v1 prefix).
    pub fn is_collection_table(table_name: &str) -> bool {
        table_name.starts_with(Self::TABLE_PREFIX)
    }

    /// SQL LIKE pattern for collection tables.
    pub fn table_pattern() -> String {
        format!("{}%", Self::TABLE_PREFIX)
    }

    /// Get the collection table prefix.
    pub fn prefix() -> &'static str {
        Self::TABLE_PREFIX
    }

    /// Name of the SDK collections metadata table.
    pub fn sdk_collections_table_name() -> &'static str {
        "sdk_collections"
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
    fn valid_names_pass() {
        let max_len = CollectionNames::MAX_COLLECTION_NAME_LENGTH;
        let valid = [
            "a",
            "A",
            "0",
            "collection_1",
            "MyCollection_123",
            &"A".repeat(max_len),
        ];
        for name in valid {
            assert!(CollectionNames::validate(name).is_ok(), "expected valid: {:?}", name);
        }
    }

    #[test]
    fn empty_name_fails() {
        let err = CollectionNames::validate("").unwrap_err();
        assert!(matches!(err, SeekDbError::InvalidInput(_)));
        let msg = err.to_string();
        assert!(msg.contains("must not be empty") || msg.contains("empty"));
    }

    #[test]
    fn name_too_long_fails() {
        let max_len = CollectionNames::MAX_COLLECTION_NAME_LENGTH;
        let long = "a".repeat(max_len + 1);
        let err = CollectionNames::validate(&long).unwrap_err();
        assert!(matches!(err, SeekDbError::InvalidInput(_)));
        let msg = err.to_string();
        assert!(msg.contains("maximum allowed") || msg.contains("too long"));
    }

    #[test]
    fn invalid_characters_fail() {
        let invalid = [
            "name-with-dash",
            "name.with.dot",
            "name with space",
            "name$",
            "名字",
        ];
        for name in invalid {
            let err = CollectionNames::validate(name).unwrap_err();
            assert!(matches!(err, SeekDbError::InvalidInput(_)), "expected invalid: {:?}", name);
            let msg = err.to_string();
            assert!(
                msg.contains("letters") || msg.contains("digits") || msg.contains("underscore") || msg.contains("a-zA-Z0-9"),
                "message for {:?}: {}",
                name,
                msg
            );
        }
    }

    #[test]
    fn table_name_and_collection_name_roundtrip() {
        let name = "my_coll";
        let table = CollectionNames::table_name(name);
        assert_eq!(table, "c$v1$my_coll");
        assert_eq!(CollectionNames::collection_name(&table), name);
    }

    #[test]
    fn table_name_v2() {
        assert_eq!(CollectionNames::table_name_v2("id123"), "c$v2$id123");
    }

    #[test]
    fn is_collection_table_and_pattern() {
        assert!(CollectionNames::is_collection_table("c$v1$foo"));
        assert!(!CollectionNames::is_collection_table("other_table"));
        assert_eq!(CollectionNames::table_pattern(), "c$v1$%");
        assert_eq!(CollectionNames::sdk_collections_table_name(), "sdk_collections");
    }
}
