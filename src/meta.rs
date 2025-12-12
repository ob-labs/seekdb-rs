/// Helpers for naming collections and columns, mirroring Python `meta_info.py`.
pub struct CollectionNames;

impl CollectionNames {
    /// Build the physical table name for a collection.
    pub fn table_name(name: &str) -> String {
        // Keep identical naming to the Python client: c$v1${collection_name}
        format!("c$v1${}", name)
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
