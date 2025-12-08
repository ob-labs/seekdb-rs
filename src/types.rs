use serde::{Deserialize, Serialize};

pub type Document = String;
pub type Documents = Vec<Document>;
pub type Embedding = Vec<f32>;
pub type Embeddings = Vec<Embedding>;
pub type Metadata = serde_json::Value;

/// Database metadata returned by admin APIs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Database {
    pub name: String,
    pub tenant: Option<String>,
    pub charset: Option<String>,
    pub collation: Option<String>,
}

/// Selects which fields to include in query/get responses.
#[derive(Clone, Copy, Debug)]
pub enum IncludeField {
    Documents,
    Metadatas,
    Embeddings,
}

/// Result shape for similarity queries (aligns with Python SDK).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryResult {
    pub ids: Vec<Vec<String>>,
    pub documents: Option<Vec<Vec<Document>>>,
    pub metadatas: Option<Vec<Vec<Metadata>>>,
    pub embeddings: Option<Vec<Vec<Embedding>>>,
    pub distances: Option<Vec<Vec<f32>>>,
}

/// Result shape for get/peek calls.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GetResult {
    pub ids: Vec<String>,
    pub documents: Option<Vec<Document>>,
    pub metadatas: Option<Vec<Metadata>>,
    pub embeddings: Option<Vec<Embedding>>,
}
