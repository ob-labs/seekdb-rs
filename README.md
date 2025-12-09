# seekdb-rs – Rust SDK for SeekDB (Server Mode)

`seekdb-rs` 是 SeekDB 的 Rust 版 SDK，当前只覆盖 **Server 模式**，通过 MySQL 协议访问 seekdb / OceanBase。整体接口与行为设计尽量对齐 Python 版 `pyseekdb`，但目前仍处于 **实验性 / 不完整** 状态。

> ⚠️ 重要说明：本 README 尽量完全对标 Python 版 README 的结构和细节，并在每个相关位置明确标出「✅ 已实现」或「❌ 未实现」的能力差异。

---

## Table of Contents

1. [Installation](#installation)
2. [Client Connection](#1-client-connection)
3. [AdminClient Connection and Database Management](#2-adminclient-connection-and-database-management)
4. [Collection (Table) Management](#3-collection-table-management)
5. [DML Operations](#4-dml-operations)
6. [DQL Operations](#5-dql-operations)
7. [Embedding Functions](#6-embedding-functions)
8. [RAG Demo](#rag-demo)
9. [Testing](#testing)
10. [Feature Matrix](#feature-matrix)

---

## Installation

Rust SDK 当前作为仓库内的子工程存在，尚未发布到 crates.io。推荐以 Workspace 本地依赖的方式使用。

```toml
# Cargo.toml (在工作区的其他 crate 中)
[dependencies]
seekdb-rs = { path = "rust-sdk" }  # 路径按实际情况调整
```

构建：

```bash
cd rust-sdk
cargo build
```

- 默认启用 `server` feature（Server 模式客户端）。
- 可选启用 `embedding` feature，集成基于 ONNX 的默认文本向量模型 `DefaultEmbedding`（依赖 `reqwest` / `tokenizers` / `ort`）。

---

## 1. Client Connection

Python 版通过统一的 `pyseekdb.Client(...)` 封装多种模式（embedded / remote server）。  
在 Rust 版中，目前仅实现了 **Server 模式客户端**：`ServerClient`。

> ✅ 对标 Python 的 `RemoteServerClient`  
> ❌ 对标 Python 的 Embedded 客户端尚未实现

### 1.1 Server Client（Remote SeekDB / OceanBase）

```rust
use seekdb_rs::{ServerClient, SeekDbError};

#[tokio::main]
async fn main() -> Result<(), SeekDbError> {
    // 使用 builder 链式配置连接参数
    let client = ServerClient::builder()
        .host("127.0.0.1") // host
        .port(2881)        // port
        .tenant("sys")     // tenant
        .database("demo")  // database
        .user("root")      // user（不含 tenant 后缀）
        .password("")      // password
        .max_connections(5)
        .build()
        .await?;

    // 执行 SQL
    let _ = client.execute("SELECT 1").await?;
    Ok(())
}
```

Rust 版 `ServerClient` 与 Python 版 `RemoteServerClient` 类似：

- 使用 `user@tenant` 的身份连接（内部自动拼接）。
- 使用 MySQL 协议访问 SeekDB / OceanBase。

### 1.2 使用环境变量构建配置

对应 Python 里从环境变量读取连接信息的用法，Rust 提供了 `ServerConfig::from_env()`：

环境变量：

- `SERVER_HOST`
- `SERVER_PORT`（默认 2881）
- `SERVER_TENANT`
- `SERVER_DATABASE`
- `SERVER_USER`
- `SERVER_PASSWORD`
- `SERVER_MAX_CONNECTIONS`（默认 5）

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
    // 从环境变量构建配置
    let config = ServerConfig::from_env()?;

    // 从配置创建客户端
    let client = ServerClient::from_config(config).await?;

    // 或者一步到位
    let client = ServerClient::from_env().await?;

    // 也可以通过 builder 从环境变量读取并覆写部分参数
    let client = ServerClient::builder()
        .from_env()?
        .database("demo_override")
        .build()
        .await?;

    client.execute("SELECT 1").await?;
    Ok(())
}
```

### 1.3 Client Methods and Properties

Rust 版中没有 Python 的统一 `Client` 工厂类，直接使用 `ServerClient`。  
主要方法：

| Method / Property                            | Status | Description                                                                 |
|----------------------------------------------|--------|-----------------------------------------------------------------------------|
| `ServerClient::builder()`                    | ✅     | 通过 builder 链式配置并创建远程客户端                                       |
| `ServerClient::from_config(ServerConfig)`    | ✅     | 从配置连接                                                                  |
| `ServerClient::from_env()`                   | ✅     | 从环境变量构建配置并连接                                                    |
| `ServerClient::pool()`                       | ✅     | 获取底层 `MySqlPool`                                                        |
| `ServerClient::tenant()` / `database()`      | ✅     | 获取当前 tenant / database                                                  |
| `ServerClient::execute(sql)`                 | ✅     | 执行不返回行的 SQL（`INSERT`/`UPDATE` 等）                                  |
| `ServerClient::fetch_all(sql)`               | ✅     | 执行查询并返回所有行                                                        |
| `ServerClient::create_collection(...)`       | ✅     | 创建 Collection（见后文）                                                   |
| `ServerClient::get_collection(...)`          | ✅     | 获取 Collection 对象                                                        |
| `ServerClient::get_or_create_collection(...)`| ✅     | 获取或创建 Collection                                                       |
| `ServerClient::delete_collection(name)`      | ✅     | 删除 Collection                                                             |
| `ServerClient::list_collections()`           | ✅     | 列出当前数据库中所有 Collection 名称                                        |
| `ServerClient::has_collection(name)`         | ✅     | 检查 Collection 是否存在                                                    |
| `ServerClient::count_collection()`           | ✅     | 统计当前数据库中 Collection 数量                                            |

> ❌ Embedded 模式（对应 Python 的 `Client(path=...)`）暂未在 Rust 中实现。

---

## 2. AdminClient Connection and Database Management

Python 版有 `AdminClient` 管理数据库（create / get / delete / list）。Rust 版提供：

- `AdminApi` trait：抽象数据库管理接口。
- `AdminClient`：基于 `ServerClient` 的薄封装。
- `ServerClient` 自身也实现了 `AdminApi`。

### 2.1 创建 AdminClient

```rust
use seekdb_rs::{AdminClient, AdminApi, ServerClient, ServerConfig, SeekDbError};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), SeekDbError> {
    let config = ServerConfig::from_env()?;
    let client = ServerClient::from_config(config).await?;

    // 使用 AdminClient（持有 Arc<ServerClient>）
    let admin = AdminClient::new(Arc::new(client));

    // 也可以直接用 ServerClient 调用同样的方法（因为实现了 AdminApi）
    // let admin: &dyn AdminApi = &client;

    Ok(())
}
```

### 2.2 Database Management APIs

接口与 Python 版语义一致：

```rust
// 创建数据库
admin.create_database("my_db", None).await?;

// 获取数据库元信息
let db = admin.get_database("my_db", None).await?;
println!("name = {}, tenant = {:?}", db.name, db.tenant);

// 列出数据库（可选 limit/offset/tenant）
let list = admin.list_databases(None, None, None).await?;

// 删除数据库
admin.delete_database("my_db", None).await?;
```

方法签名（`AdminApi`）：

```rust
#[async_trait::async_trait]
pub trait AdminApi {
    async fn create_database(&self, name: &str, tenant: Option<&str>) -> Result<()>;
    async fn get_database(&self, name: &str, tenant: Option<&str>) -> Result<Database>;
    async fn delete_database(&self, name: &str, tenant: Option<&str>) -> Result<()>;
    async fn list_databases(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        tenant: Option<&str>,
    ) -> Result<Vec<Database>>;
}
```

其中 `Database` 对应 Python 版的结构：

```rust
pub struct Database {
    pub name: String,
    pub tenant: Option<String>,
    pub charset: Option<String>,
    pub collation: Option<String>,
}
```

---

## 3. Collection (Table) Management

Rust 版 `Collection<Ef>` 对应 Python 版的 `Collection` 类。  
主要差异：

- Rust 里是参数化泛型：`Collection<Ef = Box<dyn EmbeddingFunction>>`
- 所有数据操作都是 `async fn`，返回 `Result<_, SeekDbError>`

### 3.1 Creating a Collection

在 Rust 中，创建 Collection 需要提供 `HnswConfig`，不能省略：

```rust
use seekdb_rs::{DistanceMetric, HnswConfig, ServerClient, SeekDbError};

#[tokio::main]
async fn main() -> Result<(), SeekDbError> {
    let config = seekdb_rs::ServerConfig::from_env()?;
    let client = ServerClient::from_config(config).await?;

    // 定义 HNSW 配置
    let hnsw = HnswConfig {
        dimension: 384,
        distance: DistanceMetric::Cosine,
    };

    // 创建 collection（不启用自动 embedding）
    let coll = client
        .create_collection(
            "my_collection",
            Some(hnsw),
            None::<Box<dyn EmbeddingFunction>>,
        )
        .await?;

    Ok(())
}
```

> 与 Python 不同：  
> - Python 可省略 configuration 或 embedding_function，由 SDK 推断维度。  
> - Rust 当前要求创建时提供 `HnswConfig`，否则会返回错误：  
>   `SeekDbError::Config("HnswConfig must be provided when creating a collection")`。

### 3.2 Getting a Collection

```rust
// 获取已存在的 collection，不指定 embedding_function
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

Rust 会通过 `DESCRIBE` 和 `SHOW CREATE TABLE` 解析表结构：

- 从 `embedding` 列的类型中解析向量维度（例如 `vector(384)`）。
- 从 `VECTOR INDEX ... distance=cosine` 解析距离度量。

> 如果目标表不存在或结构不符合预期，将返回 `SeekDbError::NotFound` 或 `SeekDbError::Config`。

### 3.3 Listing / Counting / Deleting Collections

```rust
// 列出所有 collection 名称
let names = client.list_collections().await?;
for n in &names {
    println!("collection = {n}");
}

// 统计 collection 数量
let count = client.count_collection().await?;

// 判断是否存在
if client.has_collection("my_collection").await? {
    println!("collection exists");
}

// 删除 collection
client.delete_collection("my_collection").await?;
```

### 3.4 Collection Properties

`Collection<Ef>` 提供以下只读属性方法：

- `name() -> &str`：Collection 名称
- `id() -> Option<&str>`：Collection ID（当前用于兼容设计，MySQL 表名仍基于 `c$v1$` 前缀）
- `dimension() -> u32`：向量维度
- `distance() -> DistanceMetric`：距离度量（L2 / Cosine / InnerProduct）
- `metadata() -> Option<&serde_json::Value>`：Collection 元数据（当前使用较少）

---

## 4. DML Operations

Rust 版已实现与 Python 版语义基本一致的 DML 操作，并且在存在 `embedding_function` 时支持 **自动 embedding**。

> ✅ 支持：显式提供 `embeddings` 的 `add/update/upsert/delete`  
> ✅ 支持：在 Collection 设置了 `embedding_function` 且仅提供 `documents` 时，`add/update/upsert` 自动生成向量（`upsert` 在无 embedding_function 时的 doc-only 调用会保留原有向量不变）

### 4.1 Add Data

`add()` 插入新的记录；若主键 `_id` 冲突会由底层数据库报错。

```rust
use seekdb_rs::{Embedding, Metadata};
use serde_json::json;

let ids = vec!["item1".to_string(), "item2".to_string()];
let embeddings: Vec<Embedding> = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
let documents = vec!["Document 1".to_string(), "Document 2".to_string()];
let metadatas: Vec<Metadata> = vec![
    json!({"category": "AI", "score": 95}),
    json!({"category": "ML", "score": 88}),
];

coll.add(&ids, Some(&embeddings), Some(&metadatas), Some(&documents))
    .await?;
```

也可以在创建 Collection 时绑定一个 `EmbeddingFunction`，只传 `documents` 让 SDK 自动生成向量：

```rust
use seekdb_rs::{DistanceMetric, HnswConfig, ServerClient, embedding::DefaultEmbedding};

let config = seekdb_rs::ServerConfig::from_env()?;
let client = ServerClient::from_config(config).await?;

// 需要在 Cargo.toml 中启用 `embedding` feature
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

// 未显式传入 embeddings，会自动调用 embedding_function 生成
coll.add(&ids, None, None, Some(&docs)).await?;
```

### 4.2 Update Data

`update()` 更新已有记录；记录必须存在，否则底层不会插入新行。

```rust
// 仅更新 metadata（允许）
coll.update(
    &["item1".to_string()],
    None, // 不更新向量
    Some(&[serde_json::json!({"category": "AI", "score": 98})]),
    None,
)
.await?;

// 同时更新 embeddings 和 documents
coll.update(
    &["item1".to_string(), "item2".to_string()],
    Some(&[vec![0.9, 0.8, 0.7], vec![0.6, 0.5, 0.4]]),
    Some(&[
        serde_json::json!({"category": "AI"}),
        serde_json::json!({"category": "ML"}),
    ]),
    Some(&[
        "Updated document 1".to_string(),
        "Updated document 2".to_string(),
    ]),
)
.await?;
```

行为：

- 若 `embeddings` 非空，会检查：
  - 长度与 `ids` 一致
  - 每个向量维度与 `Collection::dimension()` 一致
- `documents` / `metadatas` 允许为空；只对提供的字段生成 `SET` 子句。
- 若未显式提供 `embeddings`，但提供了 `documents` 且 Collection 绑定了 `embedding_function`，会自动对这些文档生成向量并更新 `embedding` 列。

### 4.3 Upsert Data

`upsert()` 在记录存在时更新，不存在时插入。支持仅更新部分字段（metadata-only / documents-only / embeddings-only）。

```rust
let id = "item1".to_string();

// 1) 首次插入
coll.upsert(
    &[id.clone()],
    Some(&[vec![1.0, 2.0, 3.0]]),
    Some(&[serde_json::json!({"tag": "init", "cnt": 1})]),
    Some(&["doc1".to_string()]),
)
.await?;

// 2) metadata-only upsert：只更新 metadata，保留 doc 和 embedding
coll.upsert(
    &[id.clone()],
    None,
    Some(&[serde_json::json!({"tag": "init", "cnt": 2})]),
    None,
)
.await?;

// 3) document-only upsert：只更新 doc
coll.upsert(
    &[id.clone()],
    None,
    None,
    Some(&["new_doc".to_string()]),
)
.await?;
```

语义（对齐 Python）：

- 入参校验：
  - `ids` 不能为空；
  - 若 `embeddings` / `documents` / `metadatas` 全部为 `None`，返回 `SeekDbError::InvalidInput`。
- 若记录已存在：
  - 仅对本次调用中提供的字段生成 `UPDATE` 语句；
  - 未提供的字段保持原值。
- 若记录不存在：
  - 插入一条新记录，缺失的字段使用默认值（`NULL` / 空数组等）。
  - 若提供了 `documents` 且 Collection 有 `embedding_function`，则会自动生成新的向量；  
    若 Collection 没有 `embedding_function`，则 document-only upsert 仅更新文档、保留原有向量。

### 4.4 Delete Data

对应 Python 的 `collection.delete(ids=..., where=..., where_document=...)`。  
Rust 使用 `Filter` / `DocFilter` 来表达条件。

```rust
use seekdb_rs::{Filter, DocFilter};
use serde_json::json;

// 按 ID 删除
coll.delete(Some(&vec!["id1".to_string(), "id2".to_string()]), None, None)
    .await?;

// 按 metadata 条件删除
let where_meta = Filter::Gte {
    field: "score".into(),
    value: json!(90),
};
coll.delete(None, Some(&where_meta), None).await?;

// 按文档全文检索条件删除
let where_doc = DocFilter::Contains("machine learning".into());
coll.delete(None, None, Some(&where_doc)).await?;
```

约束：

- 若 `ids` / `where_meta` / `where_doc` 全部为 `None`，返回  
  `SeekDbError::InvalidInput("must provide at least one of ids/where_meta/where_doc")`。

---

## 5. DQL Operations

Python 版在 `Collection` 上提供 `query`（向量搜索）、`get`（过滤读取）、`hybrid_search`。  
Rust 版对应为：

- `query_embeddings` ✅ 已实现
- `query_texts` ✅ 已实现（基于 Collection 上的 `embedding_function` 自动生成查询向量）
- `get` ✅ 已实现
- `hybrid_search` ✅ 已实现（基础能力：支持文本向量查询 + 可选 metadata / 文本过滤；复杂 search_params 需手动构造 JSON）

### 5.1 Vector Similarity Query (`query_embeddings`)

```rust
use seekdb_rs::{Filter, DocFilter, IncludeField};
use serde_json::json;

// 构造查询向量（支持批量）
let query_embeddings = vec![vec![0.0, 0.0, 0.0]];

// metadata 过滤条件
let where_meta = Filter::Gt {
    field: "score".into(),
    value: json!(10),
};

// 文本过滤条件
let where_doc = DocFilter::Contains("rust".into());

// include 控制返回字段（默认返回 documents + metadatas）
let include = &[
    IncludeField::Documents,
    IncludeField::Metadatas,
    // IncludeField::Embeddings, // 如需返回向量需显式指定
];

let result = coll
    .query_embeddings(&query_embeddings, 5, Some(&where_meta), Some(&where_doc), Some(include))
    .await?;

println!("query result ids: {:?}", result.ids);
```

返回类型 `QueryResult`：

- `ids: Vec<Vec<String>>`：一维是 query，二维是结果列表 ID。
- `documents: Option<Vec<Vec<String>>>`
- `metadatas: Option<Vec<Vec<Metadata>>>`
- `embeddings: Option<Vec<Vec<Embedding>>>`
- `distances: Option<Vec<Vec<f32>>>`

默认行为：

- 若 `include` 为 `None`：
  - 返回 `documents` + `metadatas`；
  - 不返回 `embeddings`；
  - 总是返回 `distances`。

### 5.2 Get (Retrieve by IDs or Filters)

```rust
use seekdb_rs::{Filter, DocFilter};
use serde_json::json;

// 1) 按单个 ID 获取
let got = coll
    .get(Some(&["123".to_string()]), None, None, None, None, None)
    .await?;

// 2) 按 metadata 过滤
let where_meta = Filter::Eq {
    field: "category".into(),
    value: json!("AI"),
};
let got = coll
    .get(None, Some(&where_meta), None, Some(10), None, None)
    .await?;

// 3) 按文档全文过滤
let where_doc = DocFilter::Contains("machine learning".into());
let got = coll
    .get(None, None, Some(&where_doc), Some(10), None, None)
    .await?;

// 4) 分页 + 指定 include 字段
let include = &[seekdb_rs::IncludeField::Documents, seekdb_rs::IncludeField::Metadatas];
let got = coll
    .get(None, None, None, Some(2), Some(1), Some(include))
    .await?;
```

返回类型 `GetResult`：

- `ids: Vec<String>`
- `documents: Option<Vec<String>>`
- `metadatas: Option<Vec<Metadata>>`
- `embeddings: Option<Vec<Embedding>>`

注意：

- 若 `include` 为 `None`，默认返回 `documents` + `metadatas`，不返回 `embeddings`。
- `limit` / `offset` 用于分页；若指定 `offset` 但 `limit` 为 `None`，内部会使用一个极大 `LIMIT` 以兼容 MySQL 语义。

### 5.3 Text Query (`query_texts`)

当 Collection 绑定了 `embedding_function` 时，可以直接用文本做向量检索，内部会自动生成查询向量并复用 `query_embeddings` 逻辑：

```rust
use seekdb_rs::{IncludeField, SeekDbError};

// 假设 coll 创建时已经绑定了某个 EmbeddingFunction
let qr = coll
    .query_texts(
        &["hello rust".to_string()],
        5,
        None,
        None,
        Some(&[IncludeField::Documents, IncludeField::Metadatas]),
    )
    .await?;

println!("top ids = {:?}", qr.ids[0]);
```

约束：

- 若 `texts` 为空，会返回 `SeekDbError::InvalidInput`；
- 若 Collection 没有 `embedding_function`，会返回 `SeekDbError::Embedding`，提示需要显式提供 `query_embeddings` 或设置 embedding_function；
- `embed_documents` 返回的向量个数 / 维度会被严格校验，必须与文本数量和 Collection 维度一致。

### 5.4 Hybrid Search（高层 API + 低层 API）

Rust 版提供了两层 hybrid_search 能力：

- 低层：`Collection::hybrid_search(queries, search_params, where_meta, where_doc, n_results, include)`，与 Python 的内部 `_collection_hybrid_search` 结构兼容，适合直接传入 search_parm JSON；
- 高层：`Collection::hybrid_search_advanced(query, knn, rank, n_results, include)`，与 Python 的用户侧 `collection.hybrid_search(query=..., knn=..., rank=...)` 语义一致。

#### 5.4.1 低层 `hybrid_search`（保持兼容）

Rust 版的 `hybrid_search` 提供两类用法：

1. **纯文本向量检索（简单用法）**  
   当只传入 `queries`，且 `search_params/where_meta/where_doc` 都为 `None` 时，`hybrid_search` 会退化为对 `query_texts` 的调用：

   ```rust
   use seekdb_rs::IncludeField;

   let qr = coll
       .hybrid_search(
           &["rust".to_string()],
           None,
           None,
           None,
           3,
           Some(&[IncludeField::Documents, IncludeField::Metadatas]),
       )
       .await?;
   ```

   这种场景下行为与 `query_texts` 等价，只是 API 上更贴近 Python 的 `collection.hybrid_search(query_texts=...)` 用法。

2. **自定义 search_parm 的 Hybrid**  
   当传入 `where_meta` / `where_doc` 或者显式给出 `search_params` 时，SDK 会构造 / 透传 search_parm JSON，并调用 `DBMS_HYBRID_SEARCH.GET_SQL` 生成实际 SQL 再执行：

   ```rust
   use seekdb_rs::{Filter, DocFilter, IncludeField};
   use serde_json::json;

   let where_meta = Filter::Gt {
       field: "score".into(),
       value: json!(10),
   };
   let where_doc = DocFilter::Contains("machine learning".into());

   let qr = coll
       .hybrid_search(
           &["machine learning".to_string()],
           None,                 // 也可以手动构造 search_params: Some(&json!(...))
           Some(&where_meta),
           Some(&where_doc),
           5,
           Some(&[IncludeField::Documents, IncludeField::Metadatas]),
       )
       .await?;
   ```

   更复杂场景下（显式 `rank`、直接提供 `knn.query_vector` 等），可以参考 Python 版 `_build_search_parm` 的结构，自己构造 `serde_json::Value` 传给 `search_params` 参数。

错误行为对齐 Python：

- 若既没有 `queries`，也没有任何过滤条件 / search_params，会返回  
  `SeekDbError::InvalidInput("hybrid_search requires queries, filters, or search_params")`；
- 若需要根据文本生成向量，但 Collection 没有设置 `embedding_function`，会返回 `SeekDbError::Embedding`。

#### 5.4.2 高层 `hybrid_search_advanced`（推荐，对齐 Python）

为对齐 Python 的 `collection.hybrid_search(query=..., knn=..., rank=...)`，Rust 版提供了类型化的高层 API：

```rust
use seekdb_rs::{
    collection::{HybridQuery, HybridKnn, HybridRank},
    DocFilter, Filter, IncludeField, Embedding,
};
use serde_json::json;

// 1）只做向量搜索（knn-only），等价于 query_embeddings/query_texts
let knn = HybridKnn {
    query_texts: None,
    query_embeddings: Some(vec![vec![1.0_f32, 2.0_f32, 3.0_f32]]),
    where_meta: None,
    n_results: Some(5),
};

let qr = coll
    .hybrid_search_advanced(
        None,              // query
        Some(knn),         // knn
        None,              // rank
        5,                 // 最终返回条数
        Some(&[IncludeField::Documents, IncludeField::Metadatas]),
    )
    .await?;

// 2）结合全文搜索 + 向量搜索 + RRF 排名
let query = HybridQuery {
    where_meta: Some(Filter::Eq {
        field: "category".into(),
        value: json!("AI"),
    }),
    where_doc: Some(DocFilter::Contains("machine".into())),
};

let knn = HybridKnn {
    query_texts: None,
    query_embeddings: Some(vec![vec![1.05_f32, 2.05_f32, 3.05_f32]]),
    where_meta: Some(Filter::Gte {
        field: "score".into(),
        value: json!(90),
    }),
    n_results: Some(10),
};

let rank = HybridRank::Rrf {
    rank_window_size: Some(60),
    rank_constant: Some(60),
};

let qr = coll
    .hybrid_search_advanced(
        Some(query),
        Some(knn),
        Some(rank),
        5,
        Some(&[IncludeField::Documents, IncludeField::Metadatas]),
    )
    .await?;
```

高层 API 的行为与 Python 对齐，关键点：

- `HybridQuery`：对应 Python `query` 字典：
  - `where_meta: Option<Filter>` → `query.where`；
  - `where_doc: Option<DocFilter>` → `query.where_document`；
  - 内部会构造与 `_build_query_expression` 等价的 search_parm 结构（支持纯 metadata、全文 + metadata 组合）。
- `HybridKnn`：对应 Python `knn` 字典：
  - `query_embeddings: Option<Vec<Embedding>>` 优先使用；
  - `query_texts: Option<Vec<String>>` 会通过 collection 上的 `embedding_function` 自动转为向量；
  - `where_meta: Option<Filter>` 对应 `knn.where`；
  - `n_results: Option<u32>` 对应 `knn.n_results`。
- `HybridRank`：当前支持：
  - `HybridRank::Rrf { rank_window_size, rank_constant }` → 编码为 `{"rrf": {...}}`；
  - `HybridRank::Raw(Value)` → 透传任意 rank JSON。

执行逻辑：

- **仅 knn（无 query / rank）时**：
  - 不直接调用 `DBMS_HYBRID_SEARCH`，而是退化为：
    - 有 `query_embeddings` → 调 `query_embeddings`；
    - 有 `query_texts` → 调 `query_texts`；
  - 这样可以在不同版本的 SeekDB/OceanBase 上获得稳定行为（避免 search_parm 兼容性问题）。
- **包含 query 或 rank 时**：
  - 首先尝试构造 search_parm 并调用 `DBMS_HYBRID_SEARCH.GET_SQL`；
  - 如数据库返回 `1210 (HY000): Invalid argument` 一类错误，会自动回落到“客户端近似实现”：
    - 将 `query.where` 与 `knn.where` 合并为 AND 组合；
    - 使用 `query_texts` / `query_embeddings` + `where_meta` + `where_doc` 走 SDK 自己的向量 / 过滤通路；
    - rank 参数暂不在客户端参与打分（但接口已经预留，对齐 Python 的形状）。

注意：

- 若 `query`、`knn`、`rank` 全部为空，会返回  
  `SeekDbError::InvalidInput("hybrid_search requires at least query, knn, or rank parameters")`；
- 若 `HybridKnn` 中既没有 `query_embeddings` 也没有 `query_texts`，同样会返回 `SeekDbError::InvalidInput`；
- 若需要使用 `query_texts`，Collection 必须绑定 `embedding_function`，否则返回 `SeekDbError::Embedding`。

### 5.5 Filter Operators

Python 使用 dict 组合 `$eq/$gt/$and/$or` 等操作符；Rust 使用类型安全的枚举表达式。

#### Metadata Filters (`Filter`)

对应 Python 的 `where`：

```rust
use seekdb_rs::Filter;
use serde_json::json;

// 等值： where={"category": "AI"} 或 {"category": {"$eq": "AI"}}
let f_eq = Filter::Eq {
    field: "category".into(),
    value: json!("AI"),
};

// 不等：$ne
let f_ne = Filter::Ne {
    field: "status".into(),
    value: json!("inactive"),
};

// 比较：$gt / $gte / $lt / $lte
let f_gte = Filter::Gte {
    field: "score".into(),
    value: json!(90),
};

// 集合：$in / $nin
let f_in = Filter::In {
    field: "tag".into(),
    values: vec![json!("ml"), json!("python")],
};

// 逻辑：$and / $or / $not
let f_and = Filter::And(vec![f_gte, f_in]);
let f_or = Filter::Or(vec![
    Filter::Eq {
        field: "category".into(),
        value: json!("AI"),
    },
    Filter::Eq {
        field: "category".into(),
        value: json!("ML"),
    },
]);
let f_not = Filter::Not(Box::new(Filter::Eq {
    field: "status".into(),
    value: json!("deleted"),
}));
```

内部会翻译为类似：

- `JSON_EXTRACT(metadata, '$.field') = ?`
- `JSON_EXTRACT(metadata, '$.field') >= ?`
- `JSON_EXTRACT(metadata, '$.field') IN (?, ?, ...)`
- 以及括号组合的 AND / OR / NOT。

#### Document Filters (`DocFilter`)

对应 Python 的 `where_document`：

```rust
use seekdb_rs::DocFilter;

// 全文包含：{"$contains": "rust"}
let f_contains = DocFilter::Contains("rust".into());

// 正则匹配：{"$regex": "^hello"}
let f_regex = DocFilter::Regex("^hello".into());

// 逻辑组合：$and / $or
let f_and = DocFilter::And(vec![f_contains.clone(), f_regex.clone()]);
let f_or = DocFilter::Or(vec![f_contains, f_regex]);
```

内部会翻译为：

- `MATCH(document) AGAINST (? IN NATURAL LANGUAGE MODE)`（contains）
- `document REGEXP ?`（regex）
- 使用括号组合 AND / OR。

---

## 6. Embedding Functions

Python 版有 `EmbeddingFunction` 协议以及默认的 `DefaultEmbeddingFunction`（ONNX 模型）。  
Rust 版对应提供了 `EmbeddingFunction` trait，以及在启用 `embedding` feature 时的本地 ONNX 实现 `DefaultEmbedding`。

### 6.1 `EmbeddingFunction` Trait

```rust
use seekdb_rs::{EmbeddingFunction, Embeddings, Result};

#[async_trait::async_trait]
pub trait EmbeddingFunction: Send + Sync {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings>;
    fn dimension(&self) -> usize;
}
```

要求：

1. `embed_documents`：输入 `&[String]`，输出 `Vec<Vec<f32>>`。
2. `dimension`：返回生成向量的维度，用于校验 HNSW 配置。

### 6.2 `DefaultEmbedding`（基于 ONNX 的默认实现）

在启用 `embedding` feature 时，可以使用内置的默认文本向量模型 `DefaultEmbedding`，对应 HuggingFace 上的 `sentence-transformers/all-MiniLM-L6-v2`：

```toml
[dependencies]
seekdb-rs = { path = "rust-sdk", features = ["embedding"] }
```

使用方式：

```rust
use seekdb_rs::embedding::DefaultEmbedding;

let ef = DefaultEmbedding::new()?;           // 维度固定为 384
let embs = ef
    .embed_documents(&["hello rust".to_string(), "seekdb vector".to_string()])
    .await?;
assert_eq!(embs.len(), 2);
assert_eq!(embs[0].len(), ef.dimension());
```

实现细节（与设计文档一致）：

- 首次调用 `DefaultEmbedding::new()` 时：
  - 若设置 `SEEKDB_ONNX_MODEL_DIR`，则：
    - 直接从该目录中读取
      - `SEEKDB_ONNX_MODEL_PATH`（默认 `onnx/model.onnx`）
      - `SEEKDB_ONNX_TOKENIZER_PATH`（默认 `tokenizer.json`），
    - 完全不访问网络。
  - 否则使用 Hugging Face 官方 `hf-hub` 客户端：
    - 使用 `SEEKDB_ONNX_CACHE_DIR`（或默认 `~/.cache/seekdb/onnx_models`）作为本地缓存目录；
    - 通过 `SEEKDB_ONNX_REPO_ID`（默认 `sentence-transformers/all-MiniLM-L6-v2`）和
      `SEEKDB_ONNX_REVISION`（默认 `main`）定位模型仓库；
    - 从该仓库中下载 `onnx/model.onnx` 与 `tokenizer.json` 到本地 cache，并重复利用。
  - 使用 `tokenizers` 进行分词 / 截断 / Padding，并通过 `ort` 执行 ONNX 推理；
  - 对输出做基于 `attention_mask` 的 mean pooling，得到 384 维向量。
- 后续调用会优先命中本地缓存，不再重复下载。

### 6.3 自定义 EmbeddingFunction（✅ 可自己实现）

尽管默认实现尚未完成，用户可以自行实现 `EmbeddingFunction`，例如调用外部服务或本地模型：

```rust
use seekdb_rs::{EmbeddingFunction, Embeddings, SeekDbError};

struct MyEmbedding;

#[async_trait::async_trait]
impl EmbeddingFunction for MyEmbedding {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings, SeekDbError> {
        // 示例：返回 dummy 向量，仅作占位
        let mut out = Vec::with_capacity(docs.len());
        for _ in docs {
            out.push(vec![0.0_f32; self.dimension()]);
        }
        Ok(out)
    }

    fn dimension(&self) -> usize {
        384
    }
}
```

注意事项：

1. 所有输出向量维度必须一致，并与 Collection 的 `dimension` 匹配。
2. 需要处理空输入（返回空数组）。
3. 若底层调用可能失败，应转换为 `SeekDbError::Embedding` 或 `SeekDbError::Other`。

### 6.4 在 Collection 中使用自定义 EmbeddingFunction（❌ 自动路径未打通）

理论上，Collection 支持将自定义 `EmbeddingFunction` 作为泛型参数或 trait object 传入：

```rust
let ef = MyEmbedding;

// 作为具体类型参数
let coll = client
    .create_collection::<MyEmbedding>(
        "my_collection",
        Some(HnswConfig {
            dimension: ef.dimension() as u32,
            distance: DistanceMetric::Cosine,
        }),
        Some(ef),
    )
    .await?;
```

在当前实现中，Collection 会按如下规则使用 `EmbeddingFunction`：

- `add` / `update`：
  - 若显式提供了 `embeddings`，优先使用之；
  - 否则在提供了 `documents` 且绑定了 `embedding_function` 时，会自动调用 `embed_documents` 生成向量；
  - 若既没有 `embeddings`，也没有 `documents`，会返回 `SeekDbError::InvalidInput`。
- `upsert`：
  - 若提供了 `embeddings`，逻辑同上；
  - 若只提供 `documents`：
    - 若有 `embedding_function`，自动生成新向量；
    - 若无 `embedding_function`，仅更新文档，保留原有向量。
- `query_texts` / `hybrid_search`（基于文本查询）：
  - 会使用绑定的 `embedding_function` 将文本转为查询向量；
  - 若未绑定 `embedding_function`，会返回 `SeekDbError::Embedding`，要求改用显式的 `query_embeddings` 或 search_params。

---

## RAG Demo

暂无

---

## Testing

Rust SDK 内包含：

- 单元测试：位于各模块的 `#[cfg(test)] mod tests` 中；
- 集成测试：`rust-sdk/tests/integration_server.rs`，依赖真实 SeekDB / OceanBase 实例。

运行集成测试：

```bash
cd rust-sdk

SEEKDB_INTEGRATION=1 SERVER_HOST=127.0.0.1 SERVER_PORT=2881 SERVER_TENANT=sys SERVER_DATABASE=test SERVER_USER=root SERVER_PASSWORD='' cargo test --tests
```

集成测试覆盖：

- 数据库 CRUD：`admin_database_crud`
- Collection DML：`collection_dml_roundtrip`
- `get_or_create_collection` / `count_collection`
- `upsert` 语义（metadata / document / embeddings 局部更新）
- 向量查询与过滤：`collection_query_and_filters`

---

## Feature Matrix

最后按模块对比一下与 Python 版的完成度：

| 模块 / 能力                                       | Status | 说明 |
|--------------------------------------------------|--------|------|
| 错误类型 `SeekDbError`                           | ✅ 已实现 | 对齐设计文档，统一错误返回 |
| 配置 `ServerConfig` / `HnswConfig` / `DistanceMetric` | ✅ 已实现 | 支持 `from_env`，字段与 Python 对齐 |
| 公共类型 `QueryResult` / `GetResult` / `IncludeField` / `Database` | ✅ 已实现 | 结构与 Python 返回值一致 |
| Server 连接层 `ServerClient`                     | ✅ 已实现 | `connect/from_config/from_env/execute/fetch_all` |
| Collection 管理：create/get/get_or_create/delete/list/has/count | ✅ 已实现 | 表名 `c$v1${name}`、向量列/索引结构与 Python 一致 |
| Collection DML：`add/update/upsert/delete`（显式 embeddings） | ✅ 已实现 | 语义对齐 Python，含长度和维度校验 |
| Collection DQL：`query_embeddings/get/count/peek` | ✅ 已实现 | 支持 metadata/doc 过滤与 include 字段 |
| 过滤表达式 `Filter` / `DocFilter` + SQL 生成     | ✅ 已实现 | 覆盖 `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$and/$or/$not` 和 `$contains/$regex` |
| 集成测试（Server 模式）                          | ✅ 已实现 | 需真实 SeekDB/OceanBase 环境 |
| `EmbeddingFunction` trait                        | ✅ 已实现 | 抽象已定义，可自实现 |
| 默认嵌入实现 `DefaultEmbedding` 模型加载与推理    | ✅ 已实现 | 在 `embedding` feature 下提供基于 ONNX 的 `all-MiniLM-L6-v2` 本地推理 |
| 自动 embedding：`add/update/upsert` 文本转向量   | ✅ 已实现 | Collection 绑定 `embedding_function` 且仅传 `documents` 时自动生成向量 |
| 文本查询：`Collection::query_texts`              | ✅ 已实现 | 基于 `embedding_function` 自动生成查询向量并复用 `query_embeddings` |
| Hybrid Search：`Collection::hybrid_search`       | ✅ 已实现 | 支持文本向量查询 + metadata/doc 过滤；复杂 search_params 需手动构造 |
| Embedded Client（嵌入式模式）                    | ❌ 未实现 | Rust 目前仅支持 Server 模式 |
| RAG Demo（Rust 端到端示例）                      | ❌ 未实现 | 目前仅有 Python demo |

如果你在使用过程中发现某个与 Python 版行为不一致的地方，欢迎直接对照 `src/pyseekdb` 的实现细节，一起完善 Rust SDK。 
