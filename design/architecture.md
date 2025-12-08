# seekdb-rs 架构设计概览

> 版本：0.1.0（仅 Server 模式 Rust SDK，实验性实现）  
> 目标读者：希望快速理解 `seekdb-rs` 代码结构与关键抽象的开发者

本仓库是 SeekDB 的 Rust 版 SDK，目前只覆盖 **Server 模式**，通过 MySQL 协议访问 seekdb / OceanBase，并尽量对齐 Python 版 `pyseekdb` 的语义和接口形态。

本文从架构视角介绍：

- 模块划分与整体分层
- 核心抽象（Client / Collection / Backend / Embedding / Filter / Error）
- 数据模型与 SQL 映射
- 典型调用链路与控制流
- 特性开关与可扩展性设计
- 测试与运行时依赖

---

## 1. 模块与分层架构

### 1.1 顶层视图

按职责可以大致分为四层：

1. **Public API 层（lib.rs）**
   - 对外导出统一的 SDK 接口与类型别名。
   - 将内部模块组合成稳定的公共表面：`ServerClient`、`AdminClient`、`Collection`、`Filter`、`DocFilter`、`QueryResult` 等。

2. **领域模型层（admin / collection / types / filters / meta / config）**
   - 封装「数据库 / collection / 查询 / 过滤 / 配置」等业务语义。
   - 尽量对齐 Python SDK 的方法命名和行为。

3. **基础设施抽象层（backend / embedding / error）**
   - 抽象 SQL Backend 与 Row 访问：`SqlBackend` + `BackendRow`。
   - 抽象 Embedding 生成：`EmbeddingFunction` trait。
   - 统一错误表示：`SeekDbError`。

4. **具体实现 & 依赖层（server / sqlx / ort / tokenizers / reqwest）**
   - `ServerClient` 基于 `sqlx::MySqlPool` 实现远程 Server 模式。
   - 可选的 ONNX 默认模型实现：`DefaultEmbedding`（feature = `embedding`）。

简化结构示意：

```text
+-----------------------------+
|        Public API           |
|  lib.rs (pub use ...)       |
+--------------+--------------+
               |
+--------------v--------------+
|      领域模型 (Domain)       |
| admin / collection / types  |
| filters / meta / config     |
+--------------+--------------+
               |
+--------------v--------------+
|   基础设施抽象 (Infra)      |
| backend::SqlBackend         |
| backend::BackendRow         |
| embedding::EmbeddingFunction|
| error::SeekDbError          |
+--------------+--------------+
               |
+--------------v--------------+
|   具体实现 / 外部依赖       |
| server::ServerClient        |
| sqlx / MySQL / OceanBase    |
| (可选) ort / tokenizers     |
+-----------------------------+
```

### 1.2 模块职责速览

- `src/lib.rs`  
  - 定义 crate 公共入口，`pub mod` 暴露内部模块。  
  - 通过 `pub use` 聚合常用类型与 trait，形成稳定 API 表面。

- `src/server.rs` – **Server 模式客户端**
  - 管理 `sqlx::MySqlPool`，负责连接、执行 SQL、管理 collection。
  - 通过 `connect_internal` 组装 `mysql://user@tenant:password@host:port/database` 连接字符串。
  - 提供 `create_collection / get_collection / list_collections / has_collection / delete_collection` 等集合管理接口。
  - 通过 `AdminApi` trait 实现数据库级管理（create/get/delete/list database）。
  - 实现 `SqlBackend`，为更高层提供统一的 SQL 执行接口。

- `src/admin.rs` – **数据库管理接口**
  - `AdminApi` trait 抽象数据库管理操作。
  - `AdminClient` 是一个薄封装，持有 `Arc<ServerClient>` 并实现 `AdminApi`。

- `src/collection.rs` – **Collection 领域模型**
  - 定义 `Collection<Ef>` 泛型结构，代表一张向量表（collection）。  
  - 提供 DML / DQL / Hybrid 高层 API：
    - DML：`add / update / upsert / delete / count / peek`
    - DQL：`get / query_embeddings / query_texts`
    - Hybrid：`hybrid_search / hybrid_search_advanced`
  - 处理 embedding 自动生成逻辑、参数长度校验、维度校验等。
  - 使用 `filters::build_where_clause` 统一构造 SQL WHERE 子句。
  - 通过 `BackendRow` 抽象将 SQL 行解析为 `QueryResult` / `GetResult`。

- `src/backend.rs` – **SQL Backend 抽象**
  - `BackendRow` trait：与 driver 无关的行访问接口（支持 bytes / string / f32 / i64 / index-based string）。
  - `SqlBackend` trait：异步 SQL 执行抽象（`execute` / `fetch_all` / `mode`）。
  - 为 `sqlx::mysql::MySqlRow` 提供 `BackendRow` 实现。
  - 为 `ServerClient` 提供 `SqlBackend` 实现。

- `src/embedding.rs` – **Embedding 抽象 + 默认实现**
  - `EmbeddingFunction` trait：抽象文本 → 向量的计算逻辑。
  - 为 `Box<dyn EmbeddingFunction>` 提供便利实现，便于作为默认泛型参数。
  - feature = `embedding` 时：
    - `DefaultEmbedding`：基于 ONNX 的 all-MiniLM-L6-v2 模型。
    - 负责模型文件缓存、下载、Tokenization、ONNX 推理与 mean pooling。
    - 使用阻塞 `reqwest` + 独立 OS 线程避免破坏外部 Tokio runtime。

- `src/filters.rs` – **Metadata / 文档过滤抽象**
  - `Filter`：基于 JSON metadata 的条件表达式（Eq / Lt / Gt / In / Nin / And / Or / Not 等）。
  - `DocFilter`：基于文档内容的过滤（`MATCH ... AGAINST` / `REGEXP`）。
  - `SqlWhere`：将上述 filter 编译为 `clause` + `params`，用于 DML/DQL 统一构造 WHERE。

- `src/meta.rs` – **命名约定**
  - `CollectionNames::table_name(name)`：将逻辑 collection 名转换为物理表名 `c$v1${name}`。
  - `CollectionFieldNames`：约定 `_id` / `document` / `embedding` / `metadata` 等列名。

- `src/types.rs` – **公共类型与结果结构**
  - 别名：`Document` / `Documents` / `Embedding` / `Embeddings` / `Metadata`。
  - `Database`：Admin API 返回的数据库信息。
  - `IncludeField`：控制查询结果中包含的字段（docs / metadatas / embeddings）。
  - `QueryResult` / `GetResult`：对齐 Python SDK 的结果结构。

- `src/config.rs` – **连接配置**
  - `ServerConfig`：封装连接参数（host / port / tenant / database / user / password / max_connections）。
  - `ServerConfig::from_env()`：从环境变量 `SERVER_*` 构建配置。
  - `DistanceMetric` / `HnswConfig`：控制向量维度与距离度量。

- `src/error.rs` – **统一错误类型**
  - `SeekDbError`：对外统一错误枚举（Connection / Sql / NotFound / Config / Embedding / InvalidInput / Serialization / Other）。
  - `Result<T>` 类型别名。
  - `From<sqlx::Error>` 实现，统一 SQL 错误映射。

---

## 2. 数据模型与 SQL 映射

### 2.1 Collection 表结构

`server.rs` 中的 `build_create_table_sql` 定义了 collection 的物理表结构：

- 表名：`c$v1${collection_name}`（由 `CollectionNames::table_name` 生成）。
- 核心列：
  - `_id varbinary(512) PRIMARY KEY NOT NULL`：文档主键。
  - `document text`：原始文档内容。
  - `embedding vector(dimension)`：向量列，维度由 `HnswConfig.dimension` 控制。
  - `metadata json`：任意 JSON 元数据。
- 索引：
  - `FULLTEXT INDEX idx_fts(document) WITH PARSER ik`：全文索引，支持 `MATCH ... AGAINST`。
  - `VECTOR INDEX idx_vec (embedding) with(distance=<l2|cosine|inner_product>, type=hnsw, lib=vsag)`：HNSW 向量索引。

`DistanceMetric` 枚举与 DB 的映射通过 `distance_str` 完成：

- `DistanceMetric::L2` → `"l2"`
- `DistanceMetric::Cosine` → `"cosine"`
- `DistanceMetric::InnerProduct` → `"inner_product"`

### 2.2 从表结构反推 Collection 元数据

`ServerClient::get_collection` 会：

1. 通过 `DESCRIBE table` 获取列信息，从 `embedding` 列类型（如 `vector(384)`）中解析维度：`parse_dimension`。
2. 通过 `SHOW CREATE TABLE` 获取建表 SQL，从 `distance=...` 片段中解析距离度量：`parse_distance`。
3. 将解析到的 `dimension` / `distance` 携带到 `Collection::new`。

由此确保：

- 即使 Collection 不是通过 Rust SDK 创建，只要表结构符合约定，也能在 Rust 侧恢复正确的 Collection 配置。

---

## 3. 核心抽象与交互关系

### 3.1 ServerClient 与 SqlBackend

**ServerClient** 的职责：

- 管理到底层 MySQL / OceanBase 的连接池（`MySqlPool`）。
- 对外暴露：
  - 通用 SQL 接口：`execute(sql)` / `fetch_all(sql)`;
  - Collection 管理：`create/get/get_or_create/list/has/delete/count_collection`;
  - Database 管理：`create_database / get_database / list_databases / delete_database`。

**SqlBackend trait** 的存在意义：

- 抽象「能执行 SQL / 返回行」的最小接口，为未来其他 Backend（如 embedded 引擎）留扩展点。
- 当前仅由 `ServerClient` 实现，依赖 `sqlx::mysql::MySqlRow` 的 `BackendRow` 实现。
- 主要被以下代码使用：
  - Hybrid 查询路径中，通过 `SqlBackend::execute / fetch_all` 生成并执行 `DBMS_HYBRID_SEARCH.GET_SQL`。

交互关系：

```text
Collection<Ef>
    ├─ 持有 Arc<ServerClient> 作为具体 backend
    ├─ 构造 SQL（DML / DQL / Hybrid）
    └─ 通过 ServerClient::pool() 或 SqlBackend trait 执行
```

当前 `Collection` 仍然直接依赖 `ServerClient`，`SqlBackend` 抽象更多服务于 Hybrid 相关代码；未来若引入 embedded backend，可以考虑让 `Collection` 泛型化 backend 类型。

### 3.2 Collection 与 EmbeddingFunction

`Collection<Ef>` 通过泛型参数 `Ef` 注入 embedding 行为：

- 默认类型为 `Box<dyn EmbeddingFunction>`，适合动态选择模型。
- `Ef: EmbeddingFunction + 'static`，满足异步 + 线程安全约束。

在 DML/DQL 中的使用：

- `add` / `update` / `upsert` 支持两种模式：
  1. 显式传入 `embeddings`；
  2. 未传 `embeddings` 且提供 `documents` 时：
     - 若 collection 绑定了 `embedding_function`：调用 `Ef::embed_documents` 自动生成。
     - 若未绑定：返回 `SeekDbError::InvalidInput` 或在某些 upsert 情况下保留原 embedding 不变。
- `query_texts` / `hybrid_search` 中：
  - 文本查询会利用 `embedding_function` 生成查询向量，然后复用 `query_embeddings` 路径。
  - 若缺少 `embedding_function`，返回 `SeekDbError::Embedding`，语义与 Python 对齐。

这种设计通过「纯 trait + 泛型」的方式，将模型实现完全解耦：

- SDK 自带的 `DefaultEmbedding` 只是一个可选实现（feature gated）。
- 用户可以自定义任意 `EmbeddingFunction` 实现并注入 `Collection`。

### 3.3 Filter / DocFilter 与 WHERE 子句生成

`filters.rs` 把 Python 版的 Filter 语义翻译为 Rust 枚举，并统一生成 SQL WHERE：

- `Filter`：针对 JSON metadata 的结构化条件（Eq / Lt / Gt / Lte / Gte / Ne / In / Nin / And / Or / Not）。
  - 编译为 `JSON_EXTRACT(metadata, '$.field') ...` 形式的表达式；
  - 同时收集绑定参数（`SqlWhere.params: Vec<Metadata>`）。
- `DocFilter`：针对文档内容的条件（`Contains` → FULLTEXT MATCH，`Regex` → `REGEXP`）。
- `build_where_clause(filter, doc_filter, ids)`：
  - 将 `ids` / metadata filter / doc filter 统一拼接为：
    - `WHERE _id IN (...) AND JSON_EXTRACT(...) AND MATCH(document) AGAINST (...)`
  - 同时构建 parameters 列表，供后续 `bind_metadata` 使用。

`Collection::get` / `delete` / `query_embeddings` / `hybrid_search` 均复用这一构建逻辑，保证：

- SQL 生成逻辑集中，便于扩展与测试；
- 与 Python 客户端的行为保持一致（包括 `Filter::In` / `DocFilter::Regex` 等场景）。

### 3.4 Hybrid Search 设计

Hybrid 搜索相关逻辑集中在 `collection.rs` 中，分为两层 API：

1. `hybrid_search`（偏底层，对应 Python 的 `search_params`/`search_parm`）
2. `hybrid_search_advanced`（高层类型化 API，对应 Python 的 `query/knn/rank`）

核心思想：

- SeekDB / OceanBase 提供 `DBMS_HYBRID_SEARCH.GET_SQL(table, search_parm)`：
  - 由引擎根据 `search_parm` 生成实际 SQL；
  - SDK 负责构造 `search_parm` 并执行生成的 SQL。
- Rust SDK：
  - 优先尝试使用 `DBMS_HYBRID_SEARCH`；
  - 若返回「invalid argument / 1210」等错误，则回退到客户端近似实现（在 `hybrid_search_advanced` 中通过组合已有的向量搜索 / get / filter 实现）。

重要设计点：

- `hybrid_search`：
  - 当只传 `queries` 且无 filter / search_params 时，直接退化为 `query_texts`，避免不必要的 `DBMS_HYBRID_SEARCH` 调用。
  - 当传入 `search_params` 时，视为专家模式：直接透传给 `DBMS_HYBRID_SEARCH`。

- `hybrid_search_advanced`：
  - `HybridQuery` / `HybridKnn` / `HybridRank` 把原本 JSON 结构用 Rust 类型表达，更安全也更易用。
  - 当只有 KNN（无 query / rank）时，不使用 `DBMS_HYBRID_SEARCH`，而是直接走 `query_embeddings`/`query_texts`，增强兼容性。
  - 当包含 query 或 rank 时：
    - 构造 `HybridSearchParam` → 序列化为 JSON → 赋值给 `@search_parm` → 调用 `DBMS_HYBRID_SEARCH.GET_SQL`→ 执行返回的 SQL。
    - 若发生 `invalid argument` 类错误，则执行 fallback 逻辑。

---

## 4. 典型调用链路

以下从「使用者视角」串联一些典型的调用路径，帮助理解控制流。

### 4.1 建立连接与数据库管理

1. 用户通过 `ServerConfig::from_env()` 或直接构造 `ServerConfig`：
   - 从 `SERVER_*` 环境变量读取 host / port / tenant / database / user / password / max_connections。
2. 调用 `ServerClient::from_config(config).await`：
   - 内部调用 `connect_internal` → 构造连接字符串 → 使用 `MySqlPoolOptions::new().connect(&url)` 建立连接池。
3. 通过 `AdminClient::new(Arc::new(client))` 或直接在 `ServerClient` 上调用 `AdminApi` 实现的接口：
   - `create_database` / `get_database` / `list_databases` / `delete_database`。
4. 错误统一映射为 `SeekDbError`。

### 4.2 Collection 生命周期与 DML

1. 创建 Collection：
   - 调用 `ServerClient::create_collection(name, Some(HnswConfig), embedding_function)`；
   - 内部构造 HNSW 创建表 SQL：
     - 使用 `CollectionNames::table_name` 生成表名；
     - 使用 `build_create_table_sql` 构造向量 + 文本索引；
   - 执行 `execute(sql)` 创建表；
   - 返回 `Collection::new(...)`。

2. 添加数据（以 `add` 为例）：
   - 参数校验：
     - `ids` 非空；
     - `documents` / `metadatas` 长度与 `ids` 一致（若非空）；
     - embedding dimension 与 collection dimension 一致；
   - 若未显式传入 embeddings 且提供 docs：
     - 调用 `embedding_function.embed_documents(docs)` 生成向量；
   - 将 embedding 序列化为字符串存入 `vector` 列（通过 `vector_to_string` 等辅助函数）。
   - 对每条记录执行 `INSERT INTO table (_id, document, metadata, embedding) VALUES (?, ?, ?, ?)`。

3. 更新 / Upsert / Delete：
   - `update`：对已存在记录做局部字段更新；
   - `upsert`：
     - 先 `get` 获取现有记录；
     - 决定是 insert 还是 update；
     - 合并新旧 metadata/doc/embedding（见 `merge_values`）。
   - `delete`：
     - 要求至少提供 `ids` / `where_meta` / `where_doc` 之一，否则报 `SeekDbError::InvalidInput`，防止误删全表。
     - 通过 `build_where_clause` 生成 WHERE 并执行 `DELETE FROM table WHERE ...`。

### 4.3 向量查询与 Hybrid 查询

1. `query_embeddings`：
   - 调用者提供查询向量 `&[Embedding]`。
   - 对每个 query：
     - 根据 `DistanceMetric` 选择距离函数（如 `vec_l2_distance` / `vec_cosine_distance`）。
     - 构造 SQL：`SELECT <fields>, distance(...) AS distance FROM table WHERE ... ORDER BY distance(...) LIMIT k`。
     - 执行查询并通过 `BackendRow` 抽象解析行，填充 `QueryResult`。

2. `query_texts`：
   - 调用 `embedding_function.embed_documents(texts)` 生成查询向量；
   - 调用 `query_embeddings` 复用上述路径。

3. `hybrid_search` / `hybrid_search_advanced`：
   - 根据上一节所述逻辑构造 `search_parm`，调用 `DBMS_HYBRID_SEARCH.GET_SQL`；或在部分场景退化为 vector-only 查询。
   - 对返回 SQL 再次执行并解析结果，最终返回 `QueryResult`。

---

## 5. 特性开关与可扩展性

### 5.1 Cargo features

- `default = ["server"]`
  - 当前默认只启用 server 模式客户端。
- `server`
  - 控制与远程 SeekDB / OceanBase 交互的代码（当前所有主逻辑都在此）。
- `embedding`
  - 启用默认 ONNX embedding 实现：
    - 依赖 `reqwest` / `tokenizers` / `ort`；
    - 从本地缓存或远程下载 all-MiniLM-L6-v2 模型。

### 5.2 后端扩展（未来演进方向）

当前架构已经预留了部分扩展点：

- `SqlBackend` / `BackendRow`：
  - 可以为 embedded 引擎、mock backend 等实现该 trait，而不必直接依赖 `sqlx::MySqlPool`。
  - 目前主要由 Hybrid 查询路径使用，未来可以让 `Collection` 泛型化 Backend 类型，更进一步解耦。

- `EmbeddingFunction`：
  - 用户可自定义任意 embedding 实现（本地模型、远程服务、RPC 等），只需实现 trait 即可。
  - 默认 ONNX 实现的网络与缓存逻辑全部封装在 `embedding.rs`，不会泄漏到其他模块。

---

## 6. 测试与运行时依赖

### 6.1 测试策略

- `tests/common`：
  - `load_config_for_integration()`：从环境加载 `ServerConfig`，仅在 `SEEKDB_INTEGRATION=1` 时启用真实集成测试。
  - 提供 `DummyEmbedding` 与 `ConstantEmbedding` 用于测试不同 embedding 情况。

- 集成测试文件：
  - `integration_client.rs`：客户端与 Admin 接口的连通性测试。
  - `integration_collection_dml.rs`：验证 DML、upsert 语义、防御性删除等。
  - `integration_query.rs`：查询与 filter 行为，对齐 README 示例。
  - `integration_hybrid.rs`：Hybrid 查询与 fallback 逻辑。
  - `readme_test.rs`：确保 README 中的示例编译运行。

测试策略重点：

- **本地单元测试**：针对 filters / embedding 等纯逻辑模块（不依赖外部服务）。
- **集成测试**：在真实 SeekDB/OceanBase 环境下验证行为是否与 Python SDK 对齐。

### 6.2 运行时依赖

- 必要：
  - `sqlx` + `tokio`：所有对数据库的访问都是异步的。
  - MySQL 协议兼容的 SeekDB/OceanBase 实例。

- 可选（启用 `embedding` feature 时）：
  - `reqwest`：下载 ONNX 模型。
  - `tokenizers`：文本分词与编码。
  - `ort`：ONNX Runtime 推理。
  - 模型缓存路径：
    - 默认：`$HOME/.cache/seekdb/onnx_models/all-MiniLM-L6-v2/onnx/`；
    - 可通过 `SEEKDB_ONNX_CACHE_DIR` 覆写。

---

## 7. 总结

`seekdb-rs` 的总体设计原则可以概括为：

- **对齐 Python 语义**：方法命名、参数组合、错误语义尽量与 Python 版保持一致，降低多语言切换成本。
- **分层清晰**：
  - Public API / Domain / Infra Abstraction / Concrete Impl 四层清晰分开；
  - Filter / Embedding / Backend 等横切关注通过 trait 进行抽象。
- **可扩展性**：
  - 预留了 embedded backend、定制 embedding Function 的扩展空间；
  - Hybrid 查询在引擎不支持时有客户端 fallback。
- **安全性与防御性**：
  - 严格的参数与维度校验；
  - delete 操作强制要求条件，避免误删全表；
  - 网络下载模型在独立线程中进行，避免破坏外部 runtime。

后续若扩展 embedded 模式或增加新的引擎类型，可以优先考虑：

- 扩展 `SqlBackend` 与 `BackendRow` 的实现；
- 让 `Collection` 对后端类型进行泛型化；
- 更新本文档中相关架构章节，标记不同 backend 模式下的差异。

