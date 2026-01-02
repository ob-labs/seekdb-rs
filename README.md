# seekdb-rs – Rust SDK for SeekDB (Server Mode)

> Also available in: [简体中文](README_zh-CN.md)

`seekdb-rs` is the official Rust SDK for SeekDB, currently focused on the **Server mode** and talking to SeekDB / OceanBase over the MySQL protocol.  
The APIs are designed to closely mirror the Python SDK (`pyseekdb`), but this crate is still **experimental / incomplete** and may evolve.

---

## Table of Contents

1. [Installation](#installation)
2. [Client Connection](#1-client-connection)
3. [AdminClient and Database Management](#2-adminclient-and-database-management)
4. [Collection Management](#3-collection-management)
5. [DML Operations](#4-dml-operations)
6. [DQL Operations](#5-dql-operations)
7. [Embedding Functions](#6-embedding-functions)
8. [Sync Client](#7-sync-client-optional-sync-feature)
9. [Testing](#8-testing)
10. [Feature Matrix](#9-feature-matrix)

---

## Installation

`seekdb-rs` is published on crates.io. The usual way to use it is to depend on the released crate:

```toml
# Cargo.toml in your application / workspace crate
[dependencies]
seekdb-rs = "0.1"
```

If you are hacking on the SDK in this repository, you can instead use a local path dependency:

```toml
[dependencies]
seekdb-rs = { path = "/path/to/seekdb-rs" } # adjust to where you cloned this repo
```

Build:

```bash
cargo build
```

Features:

- `server` (enabled by default): async client for the remote SeekDB / OceanBase server.
- `embedding` (enabled by default): built‑in ONNX‑based embedding implementation (`DefaultEmbedding`), depends on `reqwest` / `tokenizers` / `ort` / `hf-hub`.
- `sync` (optional): blocking wrapper around the async client (`SyncServerClient`, `SyncCollection`), backed by an internal Tokio runtime.

Example enabling `sync` and `embedding` explicitly from crates.io:

```toml
[dependencies]
seekdb-rs = { version = "0.1", features = ["server", "embedding", "sync"] }
```

---

## 1. Client Connection

The Python SDK exposes a single `Client` factory that hides embedded vs remote server.  
In Rust we currently only support the **remote server client**, represented by `ServerClient`.

> Embedded mode (equivalent to Python’s embedded client) is not implemented in Rust yet.

### 1.1 Connecting with `ServerClient`

```rust
use seekdb_rs::{ServerClient, SeekDbError};

#[tokio::main]
async fn main() -> Result<(), SeekDbError> {
    // Build a client using a fluent builder
    let client = ServerClient::builder()
        .host("127.0.0.1") // host
        .port(2881)        // port
        .tenant("sys")     // tenant
        .database("demo")  // database
        .user("root")      // user (without tenant suffix)
        .password("")      // password
        .max_connections(5)
        .build()
        .await?;

    // Run an arbitrary SQL statement
    let _ = client.execute("SELECT 1").await?;
    Ok(())
}
```

The Rust `ServerClient` behaves similarly to Python’s `RemoteServerClient`:

- Uses `user@tenant` behind the scenes to connect.
- Talks to SeekDB / OceanBase via the MySQL protocol.

### 1.2 Configuration from Environment Variables

For parity with Python’s “read config from environment variables” pattern, Rust exposes `ServerConfig::from_env()` and helpers on `ServerClient`.

Environment variables:

- `SERVER_HOST`
- `SERVER_PORT` (default: `2881`)
- `SERVER_TENANT`
- `SERVER_DATABASE`
- `SERVER_USER`
- `SERVER_PASSWORD`
- `SERVER_MAX_CONNECTIONS` (default: `5`)

```bash
export SERVER_HOST=127.0.0.1
export SERVER_PORT=2881
export SERVER_TENANT=sys
export SERVER_DATABASE=demo
export SERVER_USER=root
export SERVER_PASSWORD=your_password
```

```rust
use seekdb_rs::{ServerClient, ServerConfig, SeekDbError};

#[tokio::main]
async fn main() -> Result<(), SeekDbError> {
    // Build config from env
    let config = ServerConfig::from_env()?;

    // Connect from config
    let client = ServerClient::from_config(config).await?;

    // Or in a single step:
    let client = ServerClient::from_env().await?;

    // Or mix env defaults with manual overrides
    let client = ServerClient::builder()
        .from_env()? // prefill from env
        .database("demo_override")
        .build()
        .await?;

    client.execute("SELECT 1").await?;
    Ok(())
}
```

### 1.3 Core Methods

There is no universal “mode‑switching” `Client` type in Rust; use `ServerClient` directly.

Key async APIs:

| Method / Property                               | Description                                                              |
|-------------------------------------------------|--------------------------------------------------------------------------|
| `ServerClient::builder()`                       | Fluent builder for creating a remote client                              |
| `ServerClient::from_config(ServerConfig)`       | Connect from an explicit config                                          |
| `ServerClient::from_env()`                      | Load config from env and connect                                         |
| `ServerClient::pool()`                          | Access the underlying `MySqlPool`                                       |
| `ServerClient::tenant()` / `database()`         | Inspect current tenant / database                                        |
| `ServerClient::execute(sql)`                    | Execute a statement that does not return rows (`INSERT` / `UPDATE` /…)   |
| `ServerClient::fetch_all(sql)`                  | Execute a query and return all rows                                      |
| `ServerClient::create_collection(...)`          | Create a collection (see below)                                          |
| `ServerClient::get_collection(...)`             | Get a `Collection` handle for an existing collection                     |
| `ServerClient::get_or_create_collection(...)`   | Get or create a collection                                               |
| `ServerClient::delete_collection(name)`         | Drop a collection                                                        |
| `ServerClient::list_collections()`              | List all collection names in the current database                        |
| `ServerClient::has_collection(name)`            | Check if a collection exists                                             |
| `ServerClient::count_collection()`              | Count collections in the current database                                |

---

## 2. AdminClient and Database Management

Python exposes an `AdminClient` for managing databases. Rust provides:

- `AdminApi` trait: abstract admin interface.
- `AdminClient`: thin wrapper around `ServerClient`.
- `ServerClient` itself implements `AdminApi`, so you can call admin methods on it directly.

### 2.1 Creating an `AdminClient`

```rust
use seekdb_rs::{AdminClient, AdminApi, ServerClient, ServerConfig, SeekDbError};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), SeekDbError> {
    let config = ServerConfig::from_env()?;
    let client = ServerClient::from_config(config).await?;

    // Use a dedicated AdminClient (holds Arc<ServerClient>)
    let admin = AdminClient::new(Arc::new(client));

    // Or call the same methods directly on ServerClient
    // let admin: &dyn AdminApi = &client;

    Ok(())
}
```

### 2.2 Database CRUD APIs

```rust
// Create a database
admin.create_database("my_db", Some("sys")).await?;

// Get metadata
let db = admin.get_database("my_db", Some("sys")).await?;
println!("name = {}, tenant = {:?}", db.name, db.tenant);

// List databases (with optional limit/offset/tenant)
let list = admin.list_databases(None, None, None).await?;

// Delete database
admin.delete_database("my_db", Some("sys")).await?;
```

`Database` corresponds to the Python struct:

```rust
pub struct Database {
    pub name: String,
    pub tenant: Option<String>,
    pub charset: Option<String>,
    pub collation: Option<String>,
}
```

---

## 3. Collection Management

`Collection<Ef>` in Rust mirrors Python’s `Collection` class, with a few important differences:

- Rust uses a generic parameter: `Collection<Ef = Box<dyn EmbeddingFunction>>`.
- All operations are async and return `Result<_, SeekDbError>`.
- With the `sync` feature enabled you also get `SyncCollection<Ef>` as a blocking wrapper.

### 3.1 Creating a Collection

When creating a collection you must provide `HnswConfig` (dimension + distance metric):

```rust
use seekdb_rs::{DistanceMetric, HnswConfig, ServerClient, SeekDbError};

#[tokio::main]
async fn main() -> Result<(), SeekDbError> {
    let config = seekdb_rs::ServerConfig::from_env()?;
    let client = ServerClient::from_config(config).await?;

    let hnsw = HnswConfig {
        dimension: 384,
        distance: DistanceMetric::Cosine,
    };

    // Create a collection without automatic embeddings
    let coll = client
        .create_collection::<Box<dyn seekdb_rs::EmbeddingFunction>>(
            "my_collection",
            Some(hnsw),
            None,
        )
        .await?;

    Ok(())
}
```

If `HnswConfig` is missing, collection creation fails with:
`SeekDbError::Config("HnswConfig must be provided when creating a collection")`.

Collection names must be non-empty, use only ASCII letters/digits/underscore (`[a-zA-Z0-9_]`), and the resulting physical table name (including the `c$v1$` prefix) must not exceed 64 characters; otherwise `SeekDbError::InvalidInput` is returned before any SQL is executed.

### 3.2 Getting a Collection

```rust
let coll = client
    .get_collection::<Box<dyn seekdb_rs::EmbeddingFunction>>("my_collection", None)
    .await?;

println!(
    "Collection name = {}, dim = {}, distance = {:?}",
    coll.name(),
    coll.dimension(),
    coll.distance()
);
```

Under the hood, SeekDB is introspected using `DESCRIBE` / `SHOW CREATE TABLE`:

- Vector dimension is parsed from the embedding column type (e.g. `vector(384)`).
- Distance metric is parsed from the vector index options (e.g. `distance=cosine`).

### 3.3 Listing / Counting / Deleting Collections

```rust
// List all collections
let names = client.list_collections().await?;

// Count collections
let count = client.count_collection().await?;

// Check existence
if client.has_collection("my_collection").await? {
    println!("collection exists");
}

// Drop collection
client.delete_collection("my_collection").await?;
```

### 3.4 Collection Properties

`Collection<Ef>` exposes a few read‑only accessors:

- `name() -> &str`
- `id() -> Option<&str>` (internal ID; table name still uses `c$v1$` prefix)
- `dimension() -> u32`
- `distance() -> DistanceMetric`
- `metadata() -> Option<&serde_json::Value>`

---

## 4. DML Operations

Rust implements DML operations with semantics close to Python:

- Explicit `embeddings` are always supported.
- When a collection has an `embedding_function`, `add` / `update` / `upsert`
  can generate embeddings automatically from `documents`.

### 4.1 `add_batch` – insert new rows (recommended)

The most ergonomic way to insert data is via the builder‑style `AddBatch`
wrapper:

```rust
use seekdb_rs::{AddBatch, Embedding, Metadata};
use serde_json::json;

let ids = vec!["item1".to_string(), "item2".to_string()];
let embeddings: Vec<Embedding> = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
let documents = vec!["Document 1".to_string(), "Document 2".to_string()];
let metadatas: Vec<Metadata> = vec![
    json!({"category": "AI", "score": 95}),
    json!({"category": "ML", "score": 88}),
];

coll.add_batch(
    AddBatch::new(&ids)
        .embeddings(&embeddings)
        .documents(&documents)
        .metadatas(&metadatas),
)
.await?;
```

When the collection was created with an `EmbeddingFunction`, you can skip the
`embeddings` parameter and let the SDK embed the documents:

```rust
use seekdb_rs::{DistanceMetric, HnswConfig, ServerClient, embedding::DefaultEmbedding};

let config = seekdb_rs::ServerConfig::from_env()?;
let client = ServerClient::from_config(config).await?;

// Requires the `embedding` feature
let ef = DefaultEmbedding::new()?;
let hnsw = HnswConfig {
    dimension: ef.dimension() as u32,
    distance: DistanceMetric::Cosine,
};

let coll = client
    .create_collection::<DefaultEmbedding>("auto_emb", Some(hnsw), Some(ef))
    .await?;

let ids = vec!["auto1".to_string(), "auto2".to_string()];
let docs = vec!["hello rust".to_string(), "seekdb vector".to_string()];

// No explicit embeddings: documents are embedded automatically
coll.add_batch(AddBatch::new(&ids).documents(&docs)).await?;
```

> The lower‑level `add(&ids, embeddings, metadatas, documents)` API is still
> available, but the builder style is preferred for readability and future
> extensibility.

### 4.2 `update_batch` – update existing rows

```rust
use seekdb_rs::UpdateBatch;

// Metadata‑only update
coll.update_batch(
    UpdateBatch::new(&["item1".to_string()])
        .metadatas(&[serde_json::json!({"category": "AI", "score": 98})]),
)
.await?;

// Update embeddings + metadata + documents
coll.update_batch(
    UpdateBatch::new(&["item1".to_string(), "item2".to_string()])
        .embeddings(&[vec![0.9, 0.8, 0.7], vec![0.6, 0.5, 0.4]])
        .metadatas(&[
            serde_json::json!({"category": "AI"}),
            serde_json::json!({"category": "ML"}),
        ])
        .documents(&[
            "Updated document 1".to_string(),
            "Updated document 2".to_string(),
        ]),
)
.await?;
```

Validation rules:

- `embeddings` (if present) must match `ids` in length.
- Each embedding must have the same dimension as `Collection::dimension()`.
- `documents` / `metadatas` can be omitted; only provided fields are updated.
- If no embeddings are provided but documents are, and the collection has an
  `embedding_function`, the SDK generates embeddings automatically.

### 4.3 `upsert_batch` – insert or update

```rust
use seekdb_rs::UpsertBatch;

let id = "item1".to_string();

// 1) Insert
coll.upsert_batch(
    UpsertBatch::new(&[id.clone()])
        .embeddings(&[vec![1.0, 2.0, 3.0]])
        .metadatas(&[serde_json::json!({"tag": "init", "cnt": 1})])
        .documents(&["doc1".to_string()]),
)
.await?;

// 2) Metadata‑only upsert: keep doc and embedding
coll.upsert_batch(
    UpsertBatch::new(&[id.clone()])
        .metadatas(&[serde_json::json!({"tag": "init", "cnt": 2})]),
)
.await?;

// 3) Document‑only upsert
coll.upsert_batch(
    UpsertBatch::new(&[id.clone()]).documents(&["new_doc".to_string()]),
)
.await?;
```

Semantics:

- `ids` must be non‑empty.
- If `embeddings` / `documents` / `metadatas` are all `None`, you get
  `SeekDbError::InvalidInput`.
- When the row exists:
  - Only fields provided in the call are updated.
  - Others keep their previous values.
- When the row does not exist:
  - A new row is inserted; missing fields use default values (`NULL` / empty).
  - If only `documents` are given and the collection has an `embedding_function`,
    embeddings are generated; otherwise only the document field is updated.

### 4.4 `delete` (with `DeleteQuery`)

Rust mirrors Python’s `collection.delete(ids=..., where=..., where_document=...)` using strongly typed filters and a builder‑style `DeleteQuery`.

```rust
use seekdb_rs::{DeleteQuery, Filter, DocFilter};
use serde_json::json;

// Delete by IDs
coll.delete_query(DeleteQuery::by_ids(&["id1".to_string(), "id2".to_string()]))
    .await?;

// Delete by metadata filter
let where_meta = Filter::Gte {
    field: "score".into(),
    value: json!(90),
};
coll.delete_query(DeleteQuery::new().with_where_meta(&where_meta))
    .await?;

// Delete by document filter
let where_doc = DocFilter::Contains("machine learning".into());
coll.delete_query(DeleteQuery::new().with_where_doc(&where_doc))
    .await?;
```

---

## 5. DQL Operations

On the Python side, collections expose `query` (vector search), `get` (filtered read),
and `hybrid_search`. Rust supports the same concepts:

- `query_embeddings` – search using explicit query embeddings.
- `query_texts` – search using raw text; embeddings are computed using the
  collection’s `EmbeddingFunction`.
- `get` – filter‑only reads.
- `hybrid_search` / `hybrid_search_advanced` – hybrid vector + text + metadata search.

You will find the complete set of examples (including hybrid search and filter
operators) in the Simplified Chinese README: [`README_zh-CN.md`](README_zh-CN.md).

---

## 6. Embedding Functions

Python defines an `EmbeddingFunction` protocol and ships a default ONNX model.  
Rust mirrors this with an `EmbeddingFunction` trait and an optional `DefaultEmbedding`
implementation behind the `embedding` feature.

### 6.1 `EmbeddingFunction` trait

```rust
use seekdb_rs::{EmbeddingFunction, Embeddings, Result};

#[async_trait::async_trait]
pub trait EmbeddingFunction: Send + Sync {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings>;
    fn dimension(&self) -> usize;
}
```

You can implement this trait for your own models (local, remote, or SaaS).  
When attached to a collection, it is used for:

- `add` / `update` / `upsert` when you only pass `documents`.
- `query_texts` and text‑based `hybrid_search`.

### 6.2 Default ONNX embedding (optional)

With the `embedding` feature enabled, `seekdb-rs` provides a built‑in
`DefaultEmbedding` based on an ONNX export of a sentence‑transformers model
(similar to `all-MiniLM-L6-v2`):

```rust
use seekdb_rs::embedding::DefaultEmbedding;

let ef = DefaultEmbedding::new()?;        // model is downloaded / loaded on demand
let dim = ef.dimension();                 // e.g. 384
let embs = ef.embed_documents(&["hello".into(), "world".into()]).await?;
```

---

## 7. Sync Client (optional `sync` feature)

For codebases that cannot adopt async/await yet, the `sync` feature enables
blocking wrappers `SyncServerClient` and `SyncCollection`. They internally own
a Tokio runtime and simply `block_on` the async implementations.

```toml
[dependencies]
seekdb-rs = { path = "/path/to/seekdb-rs", features = ["sync"] }
```

```rust
use seekdb_rs::{ServerConfig, SyncServerClient, SyncCollection, SeekDbError};

fn main() -> Result<(), SeekDbError> {
    let config = ServerConfig::from_env()?;
    let client = SyncServerClient::from_config(config)?;

    let hnsw = seekdb_rs::HnswConfig {
        dimension: 3,
        distance: seekdb_rs::DistanceMetric::Cosine,
    };

    let coll: SyncCollection = client
        .create_collection::<seekdb_rs::DummyEmbedding>("sync_demo", Some(hnsw), None::<seekdb_rs::DummyEmbedding>)?;

    let ids = vec!["id1".to_string(), "id2".to_string()];
    let embs = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
    coll.add(&ids, Some(&embs), None, Some(&["doc1".into(), "doc2".into()]))?;

    let cnt = coll.count()?;
    assert_eq!(cnt, 2);

    Ok(())
}
```

> Do not call the blocking APIs (`SyncServerClient`, `SyncCollection`) from
> within an existing Tokio runtime; use them only in non‑async contexts.

---

## 8. Testing

This crate ships both unit tests and async integration tests.

- Unit tests live alongside modules under `src/` and can be run with `cargo test`.
- Integration tests live under `tests/` and require a real SeekDB / OceanBase
  instance, controlled by env variables.

Run integration tests:

```bash
SEEKDB_INTEGRATION=1 \
SERVER_HOST=127.0.0.1 \
SERVER_PORT=2881 \
SERVER_TENANT=sys \
SERVER_DATABASE=test \
SERVER_USER=root \
SERVER_PASSWORD='' \
cargo test --tests
```

Integration tests cover:

- Database CRUD and admin APIs.
- Collection DML semantics and metadata handling.
- ONNX‑based default embedding (with the `embedding` feature).
- Hybrid search behavior.
- Sync client wrappers (with the `sync` feature).

---

## 9. Feature Matrix

A high‑level comparison with the Python SDK:

| Area                                             | Status | Notes                                                                 |
|--------------------------------------------------|--------|-----------------------------------------------------------------------|
| Error type `SeekDbError`                         | ✅     | Unified error type, aligned with the design docs                      |
| Config types `ServerConfig` / `HnswConfig` / `DistanceMetric` | ✅ | Includes `from_env` helpers                                           |
| Common structs `QueryResult` / `GetResult` / `IncludeField` / `Database` | ✅ | Struct shapes match Python                                            |
| Server client `ServerClient`                     | ✅     | `connect`/`from_config`/`from_env`/`execute`/`fetch_all`              |
| Collection mgmt: create/get/get_or_create/delete/list/has/count | ✅ | Table naming, vector column/index follow Python conventions           |
| Collection DML: `add` / `update` / `upsert` / `delete` (explicit embeddings) | ✅ | Length & dimension checks; semantics aligned with Python              |
| Collection DQL: `query_embeddings` / `query_texts` / `get` / `count` / `peek` | ✅ | Supports metadata/document filters and include flags                  |
| Filter expressions `Filter` / `DocFilter`        | ✅     | Typed equivalents of `$eq/$ne/$gt/...` and `$contains/$regex`         |
| Integration tests (server mode)                  | ✅     | Require real SeekDB / OceanBase                                       |
| `EmbeddingFunction` trait                        | ✅     | Custom implementations supported                                      |
| Default embedding implementation `DefaultEmbedding` | ✅   | ONNX‑based, behind the `embedding` feature                            |
| Auto‑embedding for `add` / `update` / `upsert`   | ✅     | When a collection has an `embedding_function`                         |
| Text queries: `Collection::query_texts`          | ✅     | Uses attached `EmbeddingFunction`                                     |
| Sync wrappers: `SyncServerClient` / `SyncCollection` | ✅  | Provided behind the `sync` feature                                    |
| Hybrid search (`hybrid_search`, `hybrid_search_advanced`) | ✅ | Hybrid vector + text + metadata search                                |
| Embedded client (on‑disk, non‑server mode)       | ❌     | Not implemented in Rust yet                                           |
| RAG demo (end‑to‑end example)                    | ❌     | Only available in Python for now                                      |

For more detailed, API‑by‑API explanations (currently in Simplified Chinese),
see [`README_zh-CN.md`](README_zh-CN.md).
