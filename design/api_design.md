# seekdb-rs 接口设计（Server 模式 SDK）

> 设计基准：尽量对齐 Python 版 `pyseekdb` 的语义与行为，但采用 Rust 风格的类型系统与错误处理。  
> 覆盖范围：当前仅包含 **Server 模式**（通过 MySQL 协议访问 seekdb / OceanBase），嵌入式模式后续演进。

风险：

- 嵌入式模式（embedded）目前尚未实现。若未来要在 Rust 侧直接复用 Python 实现，可能需要通过 PyO3 调用 Python 包，需要单独调研，优先级低于 server 模式。

---

## 1. 作用范围与目标

- 支持通过 MySQL 协议访问 SeekDB / OceanBase。
- 完成能力：
  - 连接管理；
  - Collection 管理；
  - DML / DQL（向量写入、查询、过滤）；
  - 可插拔嵌入（`EmbeddingFunction` trait + `DefaultEmbedding`）；
  - 基础 Hybrid Search（含高级 `hybrid_search_advanced`）。
- 要求：
  - 对外错误统一为 `SeekDbError`（通过 `type Result<T> = std::result::Result<T, SeekDbError>` 别名暴露）；
  - 对齐 Python SDK 行为（尤其是参数检查、默认 include 字段、错误语义）。

---

## 2. 模块划分与 re-export 约定

主要模块及职责：

- `error.rs`：统一错误类型 `SeekDbError` 与 `Result<T>` 别名。
- `config.rs`：`ServerConfig`、`HnswConfig`、`DistanceMetric` 等配置类型。
- `types.rs`：`QueryResult`、`GetResult`、`IncludeField`、`Database` 等公共类型。
- `server.rs`：`ServerClient`（Server-only 客户端）。
- `admin.rs`：`AdminApi` trait 与 `AdminClient` 封装数据库管理接口。
- `collection.rs`：`Collection<Ef>`（封装向量集合的 DML/DQL/Hybrid 操作）。
- `embedding.rs`：`EmbeddingFunction` trait 与默认实现 `DefaultEmbedding`（feature = `embedding`）。
- `filters.rs`：`Filter` / `DocFilter` 抽象与 SQL WHERE 子句生成。
- `meta.rs`：Collection 表名与列名约定（`c$v1${name}` 等）。
- `backend.rs`：`SqlBackend` / `BackendRow` 抽象，用于 decouple driver 与高层逻辑。

在 `lib.rs` 中：

- 通过 `pub mod ...` 暴露上述模块；
- 通过 `pub use` 聚合常用类型（`ServerClient`、`AdminClient`、`AdminApi`、`Collection`、`ServerConfig`、`HnswConfig`、`DistanceMetric`、`Filter`、`DocFilter`、`QueryResult`、`GetResult` 等）

---

## 3. 公共错误与配置类型

### 3.1 错误类型（error.rs）

接口设计与当前实现一致：

```rust
pub type Result<T> = std::result::Result<T, SeekDbError>;

#[derive(thiserror::Error, Debug)]
pub enum SeekDbError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("sql error: {0}")]
    Sql(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
```

说明：

- 所有对外 API 均返回 `crate::error::Result<T>`，即 `Result<T, SeekDbError>`。
- `From<sqlx::Error>` 实现统一将 driver 错误转换为 `SeekDbError::Sql` 或 `SeekDbError::NotFound`。

### 3.2 配置与 HNSW 类型（config.rs）

接口设计与当前实现一致：

```rust
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub tenant: String,
    pub database: String,
    pub user: String,
    pub password: String,
    pub max_connections: u32,
}

impl ServerConfig {
    /// 从环境变量构建配置：
    /// SERVER_HOST / SERVER_PORT / SERVER_TENANT /
    /// SERVER_DATABASE / SERVER_USER / SERVER_PASSWORD /
    /// SERVER_MAX_CONNECTIONS（可选，默认 5）。
    pub fn from_env() -> Result<Self>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    L2,
    Cosine,
    InnerProduct,
}

impl DistanceMetric {
    pub fn as_str(&self) -> &'static str;
}

#[derive(Clone, Debug)]
pub struct HnswConfig {
    pub dimension: u32,
    pub distance: DistanceMetric,
}
```

---

## 4. 公共结果类型与 include 字段（types.rs）

接口设计与当前实现基本一致，仅补充 `Database`：

```rust
pub type Document = String;
pub type Documents = Vec<Document>;
pub type Embedding = Vec<f32>;
pub type Embeddings = Vec<Embedding>;
pub type Metadata = serde_json::Value;

/// Admin API 返回的数据库信息。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Database {
    pub name: String,
    pub tenant: Option<String>,
    pub charset: Option<String>,
    pub collation: Option<String>,
}

/// 查询结果中可选择包含的字段。
#[derive(Clone, Copy, Debug)]
pub enum IncludeField {
    Documents,
    Metadatas,
    Embeddings,
}

/// 向量/Hybrid 查询统一结果结构（对齐 Python SDK）。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryResult {
    pub ids: Vec<Vec<String>>,
    pub documents: Option<Vec<Vec<Document>>>,
    pub metadatas: Option<Vec<Vec<Metadata>>>,
    pub embeddings: Option<Vec<Vec<Embedding>>>,
    pub distances: Option<Vec<Vec<f32>>>,
}

/// get/peek 等接口的结果结构。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GetResult {
    pub ids: Vec<String>,
    pub documents: Option<Vec<Document>>,
    pub metadatas: Option<Vec<Metadata>>,
    pub embeddings: Option<Vec<Embedding>>,
}
```

约定：

- 向量查询默认 include 文档和 metadata，不包含 embeddings，行为与 Python 对齐。
- 多查询向量场景下 `QueryResult` 的各字段都是「二维数组」，第一维为 query 维度。

---

## 5. ServerClient 与 Admin 接口设计

### 5.1 ServerClient

当前实现：

```rust
#[derive(Clone)]
pub struct ServerClient {
    pool: sqlx::MySqlPool,
    tenant: String,
    database: String,
}

impl ServerClient {
    /// 通过参数建立连接池。
    pub async fn connect(
        host: &str,
        port: u16,
        tenant: &str,
        database: &str,
        user: &str,
        password: &str,
    ) -> Result<Self>;

    /// 从 ServerConfig 构建。
    pub async fn from_config(config: ServerConfig) -> Result<Self>;

    /// 从环境变量构建 ServerConfig 再连接。
    pub async fn from_env() -> Result<Self>;

    pub fn pool(&self) -> &sqlx::MySqlPool;
    pub fn tenant(&self) -> &str;
    pub fn database(&self) -> &str;

    /// 执行不返回行的 SQL。
    pub async fn execute(&self, sql: &str) -> Result<sqlx::mysql::MySqlQueryResult>;

    /// 执行查询并返回所有行。
    pub async fn fetch_all(&self, sql: &str) -> Result<Vec<sqlx::mysql::MySqlRow>>;
}
```

### 5.2 Collection

设计与实现（略去内部细节）：

```rust
impl ServerClient {
    pub async fn create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>>;

    pub async fn get_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>>;

    pub async fn delete_collection(&self, name: &str) -> Result<()>;
    pub async fn list_collections(&self) -> Result<Vec<String>>;
    pub async fn has_collection(&self, name: &str) -> Result<bool>;

    /// 存在则 get，不存在则 create。
    pub async fn get_or_create_collection<Ef: EmbeddingFunction + 'static>(
        &self,
        name: &str,
        config: Option<HnswConfig>,
        embedding_function: Option<Ef>,
    ) -> Result<Collection<Ef>>;

    /// 统计当前 database 下的 collection 数量。
    pub async fn count_collection(&self) -> Result<usize>;
}
```

约束与行为：

- 表名通过 `meta::CollectionNames::table_name(name)` 统一映射为 `c$v1${name}`，与 Python SDK 对齐。
- `create_collection` 要求 `config: Option<HnswConfig>` 不能为空，否则返回 `SeekDbError::Config`。
- `get_collection` 通过 `DESCRIBE` / `SHOW CREATE TABLE` 解析 `embedding` 列的维度和 `distance=` 配置。

### 5.3 AdminApi 与 AdminClient（admin.rs / server.rs）

Python 有独立的 `AdminClient`，Rust 设计中：

- 定义一个 `AdminApi` trait 抽象数据库管理能力；
- `ServerClient` 实现 `AdminApi`；
- `AdminClient` 是基于 `Arc<ServerClient>` 的封装，便于在不同上下文中共享。

接口设计与实现一致：

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

#[derive(Clone)]
pub struct AdminClient {
    inner: std::sync::Arc<ServerClient>,
}

impl AdminClient {
    pub fn new(inner: std::sync::Arc<ServerClient>) -> Self;
}
```

说明：

- `ServerClient` 也实现了 `AdminApi`，调用者既可以直接在 `ServerClient` 上调用 admin 方法，也可以通过 `AdminClient`。
- `list_databases` 返回 `Database` 结构，而不是单纯的字符串名称，保留 charset / collation 信息。

---

## 6. Collection 接口设计（collection.rs）

### 6.1 结构体与基本属性

当前实现：

```rust
#[derive(Clone)]
pub struct Collection<Ef = Box<dyn EmbeddingFunction>> {
    client: std::sync::Arc<ServerClient>,
    name: String,
    id: Option<String>,
    dimension: u32,
    distance: DistanceMetric,
    embedding_function: Option<Ef>,
    metadata: Option<serde_json::Value>,
}

impl<Ef: EmbeddingFunction + 'static> Collection<Ef> {
    pub fn new(
        client: std::sync::Arc<ServerClient>,
        name: String,
        id: Option<String>,
        dimension: u32,
        distance: DistanceMetric,
        embedding_function: Option<Ef>,
        metadata: Option<serde_json::Value>,
    ) -> Self;

    pub fn name(&self) -> &str;
    pub fn dimension(&self) -> u32;
    pub fn distance(&self) -> DistanceMetric;
    pub fn id(&self) -> Option<&str>;
    pub fn metadata(&self) -> Option<&serde_json::Value>;
}
```

说明：

- `client` 为共享的 `Arc<ServerClient>`，保证多个 Collection/Component 可复用同一连接。
- `id` / `metadata` 目前主要为保持与 Python 模型对齐，表名仍由 `c$v1$` 前缀控制。

### 6.2 DML 接口

接口设计与当前实现（简化错误类型）：

```rust
impl<Ef: EmbeddingFunction + 'static> Collection<Ef> {
    pub async fn add(
        &self,
        ids: &[String],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()>;

    pub async fn update(
        &self,
        ids: &[String],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()>;

    pub async fn upsert(
        &self,
        ids: &[String],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()>;

    pub async fn delete(
        &self,
        ids: Option<&[String]>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
    ) -> Result<()>;
}
```

语义约定（对齐实现与 Python）：

- 通用：
  - `ids` 必须非空（对 `add` / `upsert` 而言）；长度必须与 embeddings / documents / metadatas（若非空）一致，否则 `SeekDbError::InvalidInput`。
  - 所有显式提供的 embedding 都会检查维度是否等于 `Collection::dimension()`。

- `add`：
  - 若提供 `embeddings`：直接使用，并做长度/维度检查；
  - 若不提供 `embeddings` 但提供 `documents`：
    - 若 `embedding_function` 存在：自动执行 `embed_documents` 生成向量；
    - 否则返回 `SeekDbError::InvalidInput`，提示需要 embeddings 或 embedding_function。

- `update`：
  - 允许部分字段更新（仅 metadata / 仅 document / 仅 embedding 等）；
  - 若需要根据文档重新生成 embeddings，要求 `embedding_function` 存在，否则 `SeekDbError::InvalidInput`；
  - 不会为不存在的 id 插入新行。

- `upsert`：
  - 支持 metadata-only / doc-only / embedding-only upsert；
  - 行为：
    - 若 `embeddings` 显式提供：验证长度及维度后使用；
    - 若仅提供 `documents`：
      - 若存在 `embedding_function`：自动 embed；
      - 若不存在 `embedding_function`：**允许** doc-only upsert（保持已有 embedding 不变）；
    - 若既无 embeddings 又无 documents 但有 metadatas：metadata-only upsert，保留 doc/embedding。

- `delete`：
  - 至少提供 `ids` / `where_meta` / `where_doc` 之一；
  - 若三个条件都为空，则返回 `SeekDbError::InvalidInput`，防止不带 WHERE 的全表删除；
  - WHERE 条件由 `filters::build_where_clause` 统一构造。

### 6.3 DQL 与 Hybrid 接口

查询接口设计与当前实现一致：

```rust
impl<Ef: EmbeddingFunction + 'static> Collection<Ef> {
    pub async fn query_embeddings(
        &self,
        query_embeddings: &[Embedding],
        n_results: u32,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult>;

    pub async fn query_texts(
        &self,
        texts: &[String],
        n_results: u32,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult>;

    pub async fn get(
        &self,
        ids: Option<&[String]>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        limit: Option<u32>,
        offset: Option<u32>,
        include: Option<&[IncludeField]>,
    ) -> Result<GetResult>;

    pub async fn count(&self) -> Result<usize>;
    pub async fn peek(&self, limit: u32) -> Result<GetResult>;
}
```

语义：

- `query_embeddings`：
  - 要求 `query_embeddings` 非空；
  - 根据 collection 的 `DistanceMetric` 选择对应距离函数构造 SQL；
  - 默认 include documents + metadatas，不包含 embeddings，除非显式在 `include` 中传入。

- `query_texts`：
  - 要求 `embedding_function` 存在，否则返回 `SeekDbError::Embedding`；
  - 内部调用 `embedding_function.embed_documents(texts)` 得到 query 向量，再委托给 `query_embeddings`。

- `get`：
  - 支持通过 ids / metadata filter / doc filter 来获取文档；
  - `limit` / `offset` 可以控制分页；
  - `include` 控制返回字段（默认为 documents + metadatas）。

- `count`：
  - 返回 `usize` 类型计数。

- `peek`：
  - 获取前 `limit` 条记录（不带过滤条件），多用于调试与快速预览。

#### 6.3.1 Hybrid 接口：基础版与高级版

当前实现提供两层 Hybrid API：

```rust
pub async fn hybrid_search(
    &self,
    queries: &[String],
    search_params: Option<&serde_json::Value>,
    where_meta: Option<&Filter>,
    where_doc: Option<&DocFilter>,
    n_results: u32,
    include: Option<&[IncludeField]>,
) -> Result<QueryResult>;

#[derive(Clone, Debug)]
pub struct HybridQuery {
    pub where_meta: Option<Filter>,
    pub where_doc: Option<DocFilter>,
}

#[derive(Clone, Debug)]
pub struct HybridKnn {
    pub query_texts: Option<Vec<String>>,
    pub query_embeddings: Option<Vec<Embedding>>,
    pub where_meta: Option<Filter>,
    pub n_results: Option<u32>,
}

#[derive(Clone, Debug)]
pub enum HybridRank {
    Rrf {
        rank_window_size: Option<u32>,
        rank_constant: Option<u32>,
    },
    Raw(serde_json::Value),
}

pub async fn hybrid_search_advanced(
    &self,
    query: Option<HybridQuery>,
    knn: Option<HybridKnn>,
    rank: Option<HybridRank>,
    n_results: u32,
    include: Option<&[IncludeField]>,
) -> Result<QueryResult>;
```

关键行为：

- `hybrid_search`：
  - 当仅提供 `queries` 且 `search_params/where_meta/where_doc` 都为空时，退化为 `query_texts`，等价于简单向量检索；
  - 当提供 `search_params` 时，将其视为专家模式：直接传入 `DBMS_HYBRID_SEARCH.GET_SQL`，由引擎生成 SQL；
  - 当需要根据查询文本生成向量时要求 `embedding_function` 存在，否则返回 `SeekDbError::Embedding`。

- `hybrid_search_advanced`：
  - 使用 Rust 类型表达 Python 版 `query/knn/rank` 字段，内部构造 `search_parm` JSON；
  - 仅 knn（无 query / rank）时：
    - 不使用 `DBMS_HYBRID_SEARCH`，直接走 `query_embeddings` / `query_texts`，增强兼容性；
  - 包含 query 或 rank 时：
    - 优先尝试 `DBMS_HYBRID_SEARCH.GET_SQL`；
    - 若返回「invalid argument / 1210」等错误，则在客户端 fallback，通过已有向量查询与过滤近似模拟 Hybrid 行为。

---

## 7. EmbeddingFunction trait 与默认实现（embedding.rs）

### 7.1 EmbeddingFunction trait

接口设计与当前实现一致：

```rust
#[async_trait::async_trait]
pub trait EmbeddingFunction: Send + Sync {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings>;
    fn dimension(&self) -> usize;
}

/// 方便直接使用 Box<dyn EmbeddingFunction> 作为泛型参数。
#[async_trait::async_trait]
impl EmbeddingFunction for Box<dyn EmbeddingFunction> {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings> { ... }
    fn dimension(&self) -> usize { ... }
}
```

### 7.2 DefaultEmbedding（feature = "embedding"）

仅在启用 `embedding` feature 时编译：

```rust
#[cfg(feature = "embedding")]
pub struct DefaultEmbedding {
    tokenizer: tokenizers::Tokenizer,
    session: std::sync::Arc<std::sync::Mutex<ort::session::Session>>,
    max_length: usize,
}

#[cfg(feature = "embedding")]
impl DefaultEmbedding {
    pub fn new() -> Result<Self>;
}

#[cfg(feature = "embedding")]
#[async_trait::async_trait]
impl EmbeddingFunction for DefaultEmbedding {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings>;
    fn dimension(&self) -> usize { 384 }
}
```

行为与约束：

- 模型：`all-MiniLM-L6-v2`，输出维度 384；
- 模型缓存路径：
  - 默认：`$HOME/.cache/seekdb/onnx_models/all-MiniLM-L6-v2/onnx`；
  - 可通过 `SEEKDB_ONNX_CACHE_DIR` 覆盖；
- 下载端点：
  - 使用 `HF_ENDPOINT` 环境变量指定 HuggingFace 镜像端点，默认 `https://hf-mirror.com`；
- 下载逻辑：
  - 使用阻塞版 `reqwest`，在独立 OS 线程中执行，避免在已有 Tokio runtime 中嵌套 runtime；
  - 缺失文件时才触发下载，已有文件会被复用。

---

## 8. 过滤表达式与 SQL 构造（filters.rs）

### 8.1 Metadata Filter

接口设计与当前实现一致：

```rust
pub enum Filter {
    Eq { field: String, value: Metadata },
    Lt { field: String, value: Metadata },
    Gt { field: String, value: Metadata },
    Lte { field: String, value: Metadata },
    Gte { field: String, value: Metadata },
    Ne { field: String, value: Metadata },
    In { field: String, values: Vec<Metadata> },
    Nin { field: String, values: Vec<Metadata> },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}
```

映射规则（与实现一致）：

- 所有 metadata 字段访问统一映射为 `JSON_EXTRACT(metadata, '$.{field}')`；
- `And` / `Or` / `Not` 对应 SQL 中的 `(...) AND (...)` / `(...) OR (...)` / `NOT (...)`；
- `In` / `Nin` 生成 `IN (?, ?, ...)` / `NOT IN (?, ?, ...)`；
- 实际 SQL 片段与参数由 `build_meta_clause` 与 `SqlWhere` 承载。

### 8.2 文档过滤 DocFilter

当前实现的 `DocFilter`：

```rust
pub enum DocFilter {
    Contains(String),
    Regex(String),
    And(Vec<DocFilter>),
    Or(Vec<DocFilter>),
}
```

SQL 映射：

- `DocFilter::Contains(text)` → `MATCH(document) AGAINST (? IN NATURAL LANGUAGE MODE)`，依赖 FULLTEXT 索引；
- `DocFilter::Regex(pattern)` → `document REGEXP ?`；
- `And/Or` 对应 `( ... ) AND ( ... )` / `( ... ) OR ( ... )`。

### 8.3 SqlWhere 辅助结构

统一封装 WHERE 子句与绑定参数：

```rust
#[derive(Clone, Debug)]
pub struct SqlWhere {
    pub clause: String,      // 例如: "WHERE _id IN (?, ?) AND JSON_EXTRACT(...) >= ?"
    pub params: Vec<Metadata>,
}

pub fn build_where_clause(
    filter: Option<&Filter>,
    doc_filter: Option<&DocFilter>,
    ids: Option<&[String]>,
) -> SqlWhere;
```

行为：

- `ids` 非空时生成 `_id IN (?, ?, ...)` 子句，并将 id 作为 `Metadata::String` 放入 params；
- metadata / doc filter 由对应子构造函数递归展开，并合并为唯一的 `WHERE` 子句；
- 若所有条件为空，则 `clause` 为空字符串，params 为空。

---

## 9. 功能点与当前实现覆盖情况

结合上文，可以将功能拆解为若干实现点：

1. **公共类型骨架**
   - `error.rs` + `config.rs` + `types.rs`；
   - 当前实现：已完成（含 `Database`、`DistanceMetric::as_str` 等）。

2. **基础连接层**
   - `ServerClient::builder/from_config/from_env/execute/fetch_all`；
   - 当前实现：已完成，基于 `sqlx::mysql::MySqlPool`。

3. **Collection 管理与元数据**
   - `ServerClient` 上的 collection 管理方法；
   - `Collection` 结构体及其基本属性（name / id / dimension / distance / metadata）；
   - 当前实现：已完成，包含从表结构反推维度与距离的逻辑。

4. **Filter / DocFilter 与 WHERE 构造**
   - `Filter` / `DocFilter` 枚举与 `build_where_clause`；
   - 当前实现：已完成，并配套单元测试。

5. **Collection DML**
   - `add/update/upsert/delete` 行为与错误语义；
   - 当前实现：已实现并通过集成测试覆盖多种场景（自动 embedding、metadata-only upsert、防御性 delete 等）。

6. **EmbeddingFunction 与 DefaultEmbedding**
   - trait 抽象 + 默认 ONNX 实现；
   - 当前实现：在启用 `embedding` feature 时可用，并有基础单元测试做形状校验。

7. **Collection DQL 与 Hybrid**
   - `query_embeddings/query_texts/get/peek/count`；
   - `hybrid_search/hybrid_search_advanced` + Typed Hybrid 配置；
   - 当前实现：已完成，包含对 `DBMS_HYBRID_SEARCH` 的调用与 fallback 逻辑，配套集成测试。

8. **Admin 接口与数据库管理**
   - `AdminApi` trait、`AdminClient` 封装与 `ServerClient` 实现；
   - 当前实现：已完成，并在集成测试中覆盖 CRUD 流程。

9. **示例与测试**
   - README 示例与 `tests/` 下的集成测试；
   - 当前实现：已存在多组集成测试对照 README 验证行为。

---

## 10. 后续演进方向（概要）

- **Embedded 模式**：
  - 可能需要独立的 Backend 实现（实现 `SqlBackend` / `BackendRow`），避免依赖 MySQL 协议；
  - 若复用 Python 实现，可考虑通过 PyO3 嵌入 Python，但需要额外关注：
    - GIL 管理与 async 互操作；
    - 构建体积与分发；
    - 多平台兼容性。

- **Backend 抽象进一步提升**：
  - 将 `Collection` 从具体的 `ServerClient` 抽象为泛型 backend（如 `Collection<B, Ef>` where `B: SqlBackend`）；
  - 使同一高层 API 可在不同引擎/模式之间复用。

当前接口设计与实现总体一致，可作为后续演进和对外文档的基准。未来扩展新 backend 或 embedding 实现时，优先保持上述 trait 与结果类型的稳定性。 
