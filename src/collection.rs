use std::sync::Arc;

use crate::backend::{BackendRow, SqlBackend};
use crate::config::DistanceMetric;
use crate::embedding::EmbeddingFunction;
use crate::error::{Result, SeekDbError};
use crate::filters::{build_where_clause, DocFilter, Filter};
use crate::meta::CollectionNames;
use crate::server::ServerClient;
use crate::types::{Embedding, GetResult, IncludeField, Metadata, QueryResult};
use serde_json::{json, Value};

/// High-level full-text / scalar query configuration for hybrid_search.
/// Mirrors Python `Collection.hybrid_search(query=...)` semantics.
#[derive(Clone, Debug)]
pub struct HybridQuery {
    pub where_meta: Option<Filter>,
    pub where_doc: Option<DocFilter>,
}

/// High-level vector search configuration for hybrid_search.
/// Mirrors Python `Collection.hybrid_search(knn=...)` semantics.
#[derive(Clone, Debug)]
pub struct HybridKnn {
    /// Query texts to be embedded via the collection's embedding_function.
    pub query_texts: Option<Vec<String>>,
    /// Pre-computed query embeddings. If provided, takes precedence over query_texts.
    pub query_embeddings: Option<Vec<Embedding>>,
    /// Metadata filter for the KNN branch.
    pub where_meta: Option<Filter>,
    /// Number of results for the KNN branch (k); defaults to 10 when None.
    pub n_results: Option<u32>,
}

/// High-level ranking configuration for hybrid_search.
/// Mirrors Python `Collection.hybrid_search(rank=...)` semantics.
#[derive(Clone, Debug)]
pub enum HybridRank {
    /// Reciprocal Rank Fusion. Fields map to the Python `{"rrf": {...}}` dict.
    Rrf {
        rank_window_size: Option<u32>,
        rank_constant: Option<u32>,
    },
    /// Escape hatch for custom rank JSON.
    Raw(Value),
}

/// Represents a single collection/table in seekdb.
#[derive(Clone)]
pub struct Collection<Ef = Box<dyn EmbeddingFunction>> {
    client: Arc<ServerClient>,
    name: String,
    id: Option<String>,
    dimension: u32,
    distance: DistanceMetric,
    embedding_function: Option<Ef>,
    metadata: Option<serde_json::Value>,
}

impl<Ef: EmbeddingFunction + 'static> Collection<Ef> {
    pub fn new(
        client: Arc<ServerClient>,
        name: String,
        id: Option<String>,
        dimension: u32,
        distance: DistanceMetric,
        embedding_function: Option<Ef>,
        metadata: Option<serde_json::Value>,
    ) -> Self {
        Self {
            client,
            name,
            id,
            dimension,
            distance,
            embedding_function,
            metadata,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub fn distance(&self) -> DistanceMetric {
        self.distance
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn metadata(&self) -> Option<&serde_json::Value> {
        self.metadata.as_ref()
    }

    // DML
    pub async fn add(
        &self,
        ids: &[String],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Err(SeekDbError::InvalidInput("ids must not be empty".into()));
        }
        // Validate document/metadata lengths (when provided)
        if let Some(docs) = documents {
            if !docs.is_empty() && docs.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "documents length does not match ids length".into(),
                ));
            }
        }
        if let Some(metas) = metadatas {
            if !metas.is_empty() && metas.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "metadatas length does not match ids length".into(),
                ));
            }
        }

        // Determine embeddings: prefer provided, otherwise auto-generate from documents using embedding_function.
        let embeddings: Vec<Embedding> = if let Some(embs) = embeddings {
            validate_lengths(ids, embs, metadatas, documents, self.dimension)?;
            embs.to_vec()
        } else if let Some(docs) = documents {
            let ef = self.embedding_function.as_ref().ok_or_else(|| {
                SeekDbError::InvalidInput(
                    "documents provided but no embeddings and no embedding function; provide embeddings or set embedding_function"
                        .into(),
                )
            })?;
            let generated = ef.embed_documents(docs).await?;
            if generated.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(format!(
                    "embeddings length {} does not match ids length {}",
                    generated.len(),
                    ids.len()
                )));
            }
            for emb in &generated {
                if emb.len() as u32 != self.dimension {
                    return Err(SeekDbError::InvalidInput(format!(
                        "embedding dimension {} does not match collection dimension {}",
                        emb.len(),
                        self.dimension
                    )));
                }
            }
            generated
        } else {
            return Err(SeekDbError::InvalidInput(
                "either provide embeddings or provide documents with embedding_function".into(),
            ));
        };

        let table = CollectionNames::table_name(&self.name);
        let sql = format!(
            "INSERT INTO `{table}` (_id, document, metadata, embedding) VALUES (?, ?, ?, ?)"
        );

        for i in 0..ids.len() {
            let id_bytes = ids[i].as_bytes();
            let doc = documents
                .and_then(|d| d.get(i))
                .map(|s| s.as_str())
                .unwrap_or("");
            let meta = metadatas.and_then(|m| m.get(i));
            let emb = &embeddings[i];

            sqlx::query(&sql)
                .bind(id_bytes)
                .bind(doc)
                .bind(meta.map(|v| serde_json::to_string(v).unwrap_or_default()))
                .bind(vector_to_string(emb))
                .execute(self.client.pool())
                .await?;
        }

        Ok(())
    }

    pub async fn update(
        &self,
        ids: &[String],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()> {
        if embeddings.is_none() && metadatas.is_none() && documents.is_none() {
            return Err(SeekDbError::InvalidInput(
                "nothing to update: provide embeddings/documents/metadatas".into(),
            ));
        }

        // Validate lengths only for provided fields
        if let Some(docs) = documents {
            if !docs.is_empty() && docs.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "documents length does not match ids length".into(),
                ));
            }
        }
        if let Some(metas) = metadatas {
            if !metas.is_empty() && metas.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "metadatas length does not match ids length".into(),
                ));
            }
        }
        let embeddings: Option<Vec<Embedding>> = if let Some(embs) = embeddings {
            if embs.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "embeddings length does not match ids length".into(),
                ));
            }
            for emb in embs {
                if emb.len() as u32 != self.dimension {
                    return Err(SeekDbError::InvalidInput(format!(
                        "embedding dimension {} does not match collection dimension {}",
                        emb.len(),
                        self.dimension
                    )));
                }
            }
            Some(embs.to_vec())
        } else if let Some(docs) = documents {
            let ef = self.embedding_function.as_ref().ok_or_else(|| {
                SeekDbError::InvalidInput(
                    "documents provided but no embeddings and no embedding function; provide embeddings or set embedding_function"
                        .into(),
                )
            })?;
            let generated = ef.embed_documents(docs).await?;
            if generated.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "embeddings length does not match ids length".into(),
                ));
            }
            for emb in &generated {
                if emb.len() as u32 != self.dimension {
                    return Err(SeekDbError::InvalidInput(format!(
                        "embedding dimension {} does not match collection dimension {}",
                        emb.len(),
                        self.dimension
                    )));
                }
            }
            Some(generated)
        } else {
            None
        };

        let table = CollectionNames::table_name(&self.name);

        for i in 0..ids.len() {
            let mut sets: Vec<(String, String)> = Vec::new();
            if let Some(docs) = documents {
                if let Some(doc) = docs.get(i) {
                    sets.push(("document".to_string(), doc.clone()));
                }
            }
            if let Some(metas) = metadatas {
                if let Some(meta) = metas.get(i) {
                    sets.push((
                        "metadata".to_string(),
                        serde_json::to_string(meta).unwrap_or_default(),
                    ));
                }
            }
            if let Some(embs) = embeddings.as_ref() {
                if let Some(emb) = embs.get(i) {
                    sets.push(("embedding".to_string(), vector_to_string(emb)));
                }
            }

            if sets.is_empty() {
                continue;
            }

            let set_clause = sets
                .iter()
                .map(|(k, _)| format!("{k} = ?"))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!("UPDATE `{table}` SET {set_clause} WHERE _id = ?");
            let mut query = sqlx::query(&sql);
            for (_, v) in &sets {
                query = query.bind(v);
            }
            query = query.bind(ids[i].as_bytes());
            query.execute(self.client.pool()).await?;
        }

        Ok(())
    }

pub async fn upsert(
        &self,
        ids: &[String],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Metadata]>,
        documents: Option<&[String]>,
    ) -> Result<()> {
        // Mirror Python semantics:
        // - metadata-only upsert allowed
        // - Only fields provided in this call are updated; others keep existing values
        // - If a record doesn't exist, insert with provided fields (missing ones become NULL/default)

        if ids.is_empty() {
            return Err(SeekDbError::InvalidInput("ids must not be empty".into()));
        }

        if embeddings.is_none() && documents.is_none() && metadatas.is_none() {
            return Err(SeekDbError::InvalidInput(
                "Neither embeddings nor documents nor metadatas provided.".into(),
            ));
        }

        if let Some(docs) = documents {
            if !docs.is_empty() && docs.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "documents length does not match ids length".into(),
                ));
            }
        }
        if let Some(metas) = metadatas {
            if !metas.is_empty() && metas.len() != ids.len() {
                return Err(SeekDbError::InvalidInput(
                    "metadatas length does not match ids length".into(),
                ));
            }
        }
        let embeddings: Option<Vec<Embedding>> = if let Some(embs) = embeddings {
            validate_lengths(ids, embs, metadatas, documents, self.dimension)?;
            Some(embs.to_vec())
        } else if let Some(docs) = documents {
            // If there is an embedding_function, auto-generate; otherwise allow doc-only upsert keeping old embedding.
            if let Some(ef) = self.embedding_function.as_ref() {
                let generated = ef.embed_documents(docs).await?;
                if generated.len() != ids.len() {
                    return Err(SeekDbError::InvalidInput(
                        "embeddings length does not match ids length".into(),
                    ));
                }
                for emb in &generated {
                    if emb.len() as u32 != self.dimension {
                        return Err(SeekDbError::InvalidInput(format!(
                            "embedding dimension {} does not match collection dimension {}",
                            emb.len(),
                            self.dimension
                        )));
                    }
                }
                Some(generated)
            } else {
                // doc-only upsert: keep existing embedding untouched
                None
            }
        } else {
            None
        };

        let table = CollectionNames::table_name(&self.name);

        for i in 0..ids.len() {
            let id = &ids[i];

            // Fetch existing row
            let existing = self
                .get(
                    Some(&[id.clone()]),
                    None,
                    None,
                    Some(1),
                    Some(0),
                    Some(&[
                        IncludeField::Documents,
                        IncludeField::Metadatas,
                        IncludeField::Embeddings,
                    ]),
                )
                .await?;

            let exists = !existing.ids.is_empty();
            let existing_doc = existing
                .documents
                .as_ref()
                .and_then(|docs| docs.first())
                .cloned();
            let existing_meta = existing
                .metadatas
                .as_ref()
                .and_then(|ms| ms.first())
                .cloned();
            let existing_emb = existing
                .embeddings
                .as_ref()
                .and_then(|es| es.first())
                .cloned();

            let new_doc = documents.and_then(|d| d.get(i)).cloned();
            let new_meta = metadatas.and_then(|m| m.get(i)).cloned();
            let new_emb = embeddings
                .as_ref()
                .and_then(|e| e.get(i))
                .cloned();

            let (final_doc, final_meta, final_emb) = merge_values(
                existing_doc,
                existing_meta,
                existing_emb,
                new_doc,
                new_meta,
                new_emb,
            );

            if exists {
                // Update only provided fields
                let mut sets: Vec<(String, String)> = Vec::new();
                if documents.is_some() {
                    sets.push(("document".to_string(), final_doc.unwrap_or_default()));
                }
                if metadatas.is_some() {
                    sets.push((
                        "metadata".to_string(),
                        serde_json::to_string(&final_meta).unwrap_or_default(),
                    ));
                }
                if embeddings.is_some() {
                    if let Some(emb) = final_emb.as_ref() {
                        sets.push(("embedding".to_string(), vector_to_string(emb)));
                    }
                }

                if !sets.is_empty() {
                    let set_clause = sets
                        .iter()
                        .map(|(k, _)| format!("{k} = ?"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let sql = format!("UPDATE `{table}` SET {set_clause} WHERE _id = ?");
                    let mut query = sqlx::query(&sql);
                    for (_, v) in &sets {
                        query = query.bind(v);
                    }
                    query = query.bind(id.as_bytes());
                    query.execute(self.client.pool()).await?;
                }
            } else {
                // Insert new row
                let sql = format!(
                    "INSERT INTO `{table}` (_id, document, metadata, embedding) VALUES (?, ?, ?, ?)"
                );
                sqlx::query(&sql)
                    .bind(id.as_bytes())
                    .bind(final_doc.unwrap_or_default())
                    .bind(serde_json::to_string(&final_meta).unwrap_or_default())
                    .bind(
                        final_emb
                            .as_ref()
                            .map(vector_to_string)
                            .unwrap_or_else(|| "[]".into()),
                    )
                    .execute(self.client.pool())
                    .await?;
            }
        }

        Ok(())
    }

    pub async fn delete(
        &self,
        ids: Option<&[String]>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
    ) -> Result<()> {
        if ids.is_none() && where_meta.is_none() && where_doc.is_none() {
            return Err(SeekDbError::InvalidInput(
                "must provide at least one of ids/where_meta/where_doc".into(),
            ));
        }

        let table = CollectionNames::table_name(&self.name);
        let sql_where = build_where_clause(where_meta, where_doc, ids);
        let sql = format!("DELETE FROM `{table}` {}", sql_where.clause);
        let mut query = sqlx::query(&sql);
        for p in sql_where.params {
            query = bind_metadata(query, &p);
        }
        query.execute(self.client.pool()).await?;
        Ok(())
    }

    // DQL
    pub async fn query_embeddings(
        &self,
        query_embeddings: &[Embedding],
        n_results: u32,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        if query_embeddings.is_empty() {
            return Err(SeekDbError::InvalidInput(
                "query_embeddings cannot be empty".into(),
            ));
        }

        let table = CollectionNames::table_name(&self.name);
        let sql_where = build_where_clause(where_meta, where_doc, None);
        let select_clause = build_select_clause(include);

        let mut all_ids = Vec::new();
        let mut all_docs = Vec::new();
        let mut all_metas = Vec::new();
        let mut all_embs = Vec::new();
        let mut all_dists = Vec::new();

        for emb in query_embeddings {
            let distance_func = distance_fn(self.distance);
            let vector_str = vector_to_string(emb);
            let sql = format!(
                "SELECT {select_clause}, {distance_func}(embedding, '{vector_str}') AS distance \
                 FROM `{table}` {where_clause} \
                 ORDER BY {distance_func}(embedding, '{vector_str}') \
                 LIMIT {limit}",
                where_clause = sql_where.clause,
                limit = n_results
            );

            let mut query = sqlx::query(&sql);
            for p in &sql_where.params {
                query = bind_metadata(query, p);
            }
            let rows = query.fetch_all(self.client.pool()).await?;

            let mut ids = Vec::new();
            let mut docs = Vec::new();
            let mut metas = Vec::new();
            let mut embs = Vec::new();
            let mut dists = Vec::new();

            for row in rows {
                ids.push(id_from_row(&row));
                if include_documents(include) {
                    let doc = row
                        .get_string("document")
                        .unwrap_or(None)
                        .unwrap_or_default();
                    docs.push(doc);
                }
                if include_metadatas(include) {
                    metas.push(metadata_from_row(&row));
                }
                if include_embeddings(include) {
                    if let Some(v) = row.get_string("embedding").unwrap_or(None) {
                        embs.push(parse_vector_string(v));
                    }
                }
                let dist = row
                    .get_f32("distance")
                    .unwrap_or(None)
                    .unwrap_or(0.0);
                dists.push(dist);
            }

            all_ids.push(ids);
            all_dists.push(dists);
            if include_documents(include) {
                all_docs.push(docs);
            }
            if include_metadatas(include) {
                all_metas.push(metas);
            }
            if include_embeddings(include) {
                all_embs.push(embs);
            }
        }

        Ok(QueryResult {
            ids: all_ids,
            documents: if include_documents(include) {
                Some(all_docs)
            } else {
                None
            },
            metadatas: if include_metadatas(include) {
                Some(all_metas)
            } else {
                None
            },
            embeddings: if include_embeddings(include) {
                Some(all_embs)
            } else {
                None
            },
            distances: Some(all_dists),
        })
    }

    pub async fn query_texts(
        &self,
        texts: &[String],
        n_results: u32,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        if texts.is_empty() {
            return Err(SeekDbError::InvalidInput(
                "texts must not be empty".into(),
            ));
        }

        let ef = self.embedding_function.as_ref().ok_or_else(|| {
            SeekDbError::Embedding(
                "Text embedding is not implemented. Provide query_embeddings directly or set embedding_function on collection.".into(),
            )
        })?;

        let embeddings = ef.embed_documents(texts).await?;
        if embeddings.len() != texts.len() {
            return Err(SeekDbError::InvalidInput(format!(
                "embeddings length {} does not match texts length {}",
                embeddings.len(),
                texts.len()
            )));
        }
        for emb in &embeddings {
            if emb.len() as u32 != self.dimension {
                return Err(SeekDbError::InvalidInput(format!(
                    "embedding dimension {} does not match collection dimension {}",
                    emb.len(),
                    self.dimension
                )));
            }
        }

        self.query_embeddings(&embeddings, n_results, where_meta, where_doc, include)
            .await
    }

    /// Hybrid search combining vector and keyword/term filters.
    pub async fn hybrid_search(
        &self,
        queries: &[String],
        search_params: Option<&serde_json::Value>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        n_results: u32,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        // Fast-path: pure vector search with text queries and no explicit search_params/filters.
        // Delegate to `query_texts` so we reuse the standard vector search path instead of
        // going through DBMS_HYBRID_SEARCH, which is primarily for true hybrid scenarios.
        if search_params.is_none()
            && where_meta.is_none()
            && where_doc.is_none()
            && !queries.is_empty()
        {
            return self
                .query_texts(queries, n_results, where_meta, where_doc, include)
                .await;
        }

        let search_parm_json = if let Some(sp) = search_params {
            sp.to_string()
        } else {
            build_search_parm_json(self, queries, where_meta, where_doc, n_results).await?
        };

        if std::env::var("DEBUG_HYBRID").is_ok() {
            eprintln!("DEBUG_HYBRID search_parm_json: {search_parm_json}");
        }

        if search_parm_json.is_empty() {
            return Err(SeekDbError::InvalidInput(
                "hybrid_search requires queries, filters, or search_params".into(),
            ));
        }

        self.execute_hybrid_search(search_parm_json, include).await
    }

    /// High-level hybrid search API mirroring Python's `Collection.hybrid_search(query=..., knn=..., rank=...)`.
    /// This builds a structured `search_parm` from typed parameters and delegates to DBMS_HYBRID_SEARCH.
    pub async fn hybrid_search_advanced(
        &self,
        query: Option<HybridQuery>,
        knn: Option<HybridKnn>,
        rank: Option<HybridRank>,
        n_results: u32,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        // Fast-path: KNN-only hybrid search – delegate to existing vector search APIs
        // instead of going through DBMS_HYBRID_SEARCH. This mirrors Python's knn-only
        // semantics while avoiding engine-specific search_parm requirements.
        if query.is_none() && rank.is_none() {
            if let Some(knn_cfg) = knn.as_ref() {
                return self
                    .hybrid_search_advanced_knn_only(knn_cfg, n_results, include)
                    .await;
            } else {
                return Err(SeekDbError::InvalidInput(
                    "hybrid_search requires at least query or knn parameters".into(),
                ));
            }
        }

        let search_parm_json =
            build_search_parm_from_typed(self, query.as_ref(), knn.as_ref(), rank.as_ref(), n_results)
                .await?;

        if std::env::var("DEBUG_HYBRID").is_ok() {
            eprintln!("DEBUG_HYBRID search_parm_json (advanced): {search_parm_json}");
        }

        if search_parm_json.is_empty() {
            return Err(SeekDbError::InvalidInput(
                "hybrid_search requires at least query, knn, or rank parameters".into(),
            ));
        }

        match self.execute_hybrid_search(search_parm_json, include).await {
            Ok(qr) => Ok(qr),
            Err(err) => {
                if is_hybrid_invalid_argument(&err) {
                    // Fallback: approximate hybrid behavior on the client side by combining
                    // filters from query/knn and delegating to existing query_texts/query_embeddings/get.
                    self.hybrid_search_advanced_fallback(query.as_ref(), knn.as_ref(), n_results, include)
                        .await
                } else {
                    Err(err)
                }
            }
        }
    }

    async fn execute_hybrid_search(
        &self,
        search_parm_json: String,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        let table = CollectionNames::table_name(&self.name);
        let escaped = search_parm_json.replace('\'', "''");
        let set_sql = format!("SET @search_parm = '{escaped}'");
        SqlBackend::execute(&*self.client, &set_sql).await?;

        let get_sql = format!(
            "SELECT DBMS_HYBRID_SEARCH.GET_SQL('{table}', @search_parm) AS query_sql FROM dual"
        );
        let rows = SqlBackend::fetch_all(&*self.client, &get_sql).await?;
        if rows.is_empty() {
            return Ok(empty_query_result(include));
        }

        let first_row = &rows[0];
        let raw_query_sql = first_row
            .get_string("query_sql")
            .unwrap_or(None)
            .or_else(|| first_row.get_string_by_index(0).unwrap_or(None))
            .unwrap_or_default();
        let query_sql = raw_query_sql.trim_matches(['\'', '"']).to_string();
        if query_sql.is_empty() {
            return Ok(empty_query_result(include));
        }

        let result_rows = SqlBackend::fetch_all(&*self.client, &query_sql).await?;
        Ok(transform_hybrid_rows(result_rows, include))
    }

    async fn hybrid_search_advanced_knn_only(
        &self,
        knn: &HybridKnn,
        n_results: u32,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        if let Some(embs) = &knn.query_embeddings {
            if embs.is_empty() {
                return Err(SeekDbError::InvalidInput(
                    "knn.query_embeddings must not be empty".into(),
                ));
            }
            let where_meta = knn.where_meta.as_ref();
            return self
                .query_embeddings(embs, n_results, where_meta, None, include)
                .await;
        }

        if let Some(texts) = &knn.query_texts {
            if texts.is_empty() {
                return Err(SeekDbError::InvalidInput(
                    "knn.query_texts must not be empty".into(),
                ));
            }
            let where_meta = knn.where_meta.as_ref();
            return self
                .query_texts(texts, n_results, where_meta, None, include)
                .await;
        }

        Err(SeekDbError::InvalidInput(
            "knn requires either query_embeddings or query_texts".into(),
        ))
    }

    async fn hybrid_search_advanced_fallback(
        &self,
        query: Option<&HybridQuery>,
        knn: Option<&HybridKnn>,
        n_results: u32,
        include: Option<&[IncludeField]>,
    ) -> Result<QueryResult> {
        // If we have a KNN branch, treat hybrid search as vector search constrained by
        // the combined filters from query.where/knn.where and query.where_document.
        if let Some(knn_cfg) = knn {
            let combined_meta = combine_meta_filters(
                query.and_then(|q| q.where_meta.as_ref()),
                knn_cfg.where_meta.as_ref(),
            );
            let where_meta = combined_meta.as_ref();
            let where_doc = query.and_then(|q| q.where_doc.as_ref());

            if let Some(embs) = &knn_cfg.query_embeddings {
                if embs.is_empty() {
                    return Err(SeekDbError::InvalidInput(
                        "knn.query_embeddings must not be empty".into(),
                    ));
                }
                return self
                    .query_embeddings(embs, n_results, where_meta, where_doc, include)
                    .await;
            }

            if let Some(texts) = &knn_cfg.query_texts {
                if texts.is_empty() {
                    return Err(SeekDbError::InvalidInput(
                        "knn.query_texts must not be empty".into(),
                    ));
                }
                return self
                    .query_texts(texts, n_results, where_meta, where_doc, include)
                    .await;
            }

            return Err(SeekDbError::InvalidInput(
                "knn requires either query_embeddings or query_texts".into(),
            ));
        }

        // No knn branch; fall back to a filter-only get() and wrap into QueryResult.
        if let Some(q) = query {
            let where_meta = q.where_meta.as_ref();
            let where_doc = q.where_doc.as_ref();
            let get_res = self
                .get(None, where_meta, where_doc, Some(n_results), Some(0), include)
                .await?;

            let num = get_res.ids.len();
            let distances = Some(vec![vec![0.0_f32; num]]);

            return Ok(QueryResult {
                ids: vec![get_res.ids],
                documents: get_res.documents.map(|d| vec![d]),
                metadatas: get_res.metadatas.map(|m| vec![m]),
                embeddings: get_res.embeddings.map(|e| vec![e]),
                distances,
            });
        }

        Err(SeekDbError::InvalidInput(
            "hybrid_search requires at least query or knn parameters".into(),
        ))
    }

    pub async fn get(
        &self,
        ids: Option<&[String]>,
        where_meta: Option<&Filter>,
        where_doc: Option<&DocFilter>,
        limit: Option<u32>,
        offset: Option<u32>,
        include: Option<&[IncludeField]>,
    ) -> Result<GetResult> {
        let table = CollectionNames::table_name(&self.name);
        let sql_where = build_where_clause(where_meta, where_doc, ids);
        let select_clause = build_select_clause(include);
        let mut sql = format!("SELECT {select_clause} FROM `{table}` {}", sql_where.clause);
        if let Some(limit) = limit {
            sql.push_str(&format!(" LIMIT {limit}"));
        }
        if let Some(offset) = offset {
            if limit.is_none() {
                sql.push_str(" LIMIT 18446744073709551615");
            }
            sql.push_str(&format!(" OFFSET {offset}"));
        }

        let mut query = sqlx::query(&sql);
        for p in &sql_where.params {
            query = bind_metadata(query, p);
        }
        let rows = query.fetch_all(self.client.pool()).await?;

        let mut result = GetResult {
            ids: Vec::new(),
            documents: if include_documents(include) {
                Some(Vec::new())
            } else {
                None
            },
            metadatas: if include_metadatas(include) {
                Some(Vec::new())
            } else {
                None
            },
            embeddings: if include_embeddings(include) {
                Some(Vec::new())
            } else {
                None
            },
        };

        for row in rows {
            result.ids.push(id_from_row(&row));
            if let Some(docs) = result.documents.as_mut() {
                let doc = row
                    .get_string("document")
                    .unwrap_or(None)
                    .unwrap_or_default();
                docs.push(doc);
            }
            if let Some(metas) = result.metadatas.as_mut() {
                metas.push(metadata_from_row(&row));
            }
            if let Some(embs) = result.embeddings.as_mut() {
                let emb = row
                    .get_string("embedding")
                    .unwrap_or(None)
                    .map(parse_vector_string)
                    .unwrap_or_default();
                embs.push(emb);
            }
        }

        Ok(result)
    }

    pub async fn count(&self) -> Result<u64> {
        let table = CollectionNames::table_name(&self.name);
        let sql = format!("SELECT COUNT(*) as cnt FROM `{table}`");
        let row = sqlx::query(&sql).fetch_one(self.client.pool()).await?;
        let cnt = row.get_i64("cnt").unwrap_or(Some(0)).unwrap_or(0);
        Ok(cnt as u64)
    }

    pub async fn peek(&self, _limit: u32) -> Result<GetResult> {
        self.get(
            None,
            None,
            None,
            Some(_limit),
            Some(0),
            Some(&[
                IncludeField::Documents,
                IncludeField::Metadatas,
                IncludeField::Embeddings,
            ]),
        )
        .await
    }
}

fn validate_lengths(
    ids: &[String],
    embeddings: &[Embedding],
    metadatas: Option<&[Metadata]>,
    documents: Option<&[String]>,
    dimension: u32,
) -> Result<()> {
    if !embeddings.is_empty() && embeddings.len() != ids.len() {
        return Err(SeekDbError::InvalidInput(format!(
            "embeddings length {} does not match ids length {}",
            embeddings.len(),
            ids.len()
        )));
    }
    for emb in embeddings {
        if emb.len() as u32 != dimension {
            return Err(SeekDbError::InvalidInput(format!(
                "embedding dimension {} does not match collection dimension {}",
                emb.len(),
                dimension
            )));
        }
    }
    if let Some(docs) = documents {
        if !docs.is_empty() && docs.len() != ids.len() {
            return Err(SeekDbError::InvalidInput(
                "documents length does not match ids length".into(),
            ));
        }
    }
    if let Some(metas) = metadatas {
        if !metas.is_empty() && metas.len() != ids.len() {
            return Err(SeekDbError::InvalidInput(
                "metadatas length does not match ids length".into(),
            ));
        }
    }
    Ok(())
}

fn vector_to_string(v: &Embedding) -> String {
    let inner = v
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

fn parse_vector_string(s: String) -> Embedding {
    s.trim_matches(&['[', ']'][..])
        .split(',')
        .filter_map(|x| x.trim().parse::<f32>().ok())
        .collect()
}

fn distance_fn(distance: DistanceMetric) -> &'static str {
    match distance {
        DistanceMetric::L2 => "l2_distance",
        DistanceMetric::Cosine => "cosine_distance",
        DistanceMetric::InnerProduct => "inner_product",
    }
}

fn build_select_clause(include: Option<&[IncludeField]>) -> String {
    let mut fields = vec!["_id".to_string()];
    if include_documents(include) {
        fields.push("document".to_string());
    }
    if include_metadatas(include) {
        // Cast metadata JSON to CHAR so that SQLx can decode it as String consistently.
        fields.push("CAST(metadata AS CHAR) AS metadata".to_string());
    }
    if include_embeddings(include) {
        fields.push("embedding".to_string());
    }
    fields.join(", ")
}

fn include_documents(include: Option<&[IncludeField]>) -> bool {
    match include {
        None => true,
        Some(list) => list.iter().any(|f| matches!(f, IncludeField::Documents)),
    }
}

fn include_metadatas(include: Option<&[IncludeField]>) -> bool {
    match include {
        None => true,
        Some(list) => list.iter().any(|f| matches!(f, IncludeField::Metadatas)),
    }
}

fn include_embeddings(include: Option<&[IncludeField]>) -> bool {
    match include {
        None => false,
        Some(list) => list.iter().any(|f| matches!(f, IncludeField::Embeddings)),
    }
}

fn id_from_row<R: BackendRow>(row: &R) -> String {
    if let Ok(Some(bytes)) = row.get_bytes("_id") {
        String::from_utf8_lossy(&bytes).into_owned()
    } else if let Ok(Some(s)) = row.get_string("_id") {
        s
    } else {
        String::new()
    }
}

fn bind_metadata<'q>(
    query: sqlx::query::Query<'q, sqlx::MySql, sqlx::mysql::MySqlArguments>,
    value: &Value,
) -> sqlx::query::Query<'q, sqlx::MySql, sqlx::mysql::MySqlArguments> {
    match value {
        Value::String(s) => query.bind(s.clone()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                query.bind(i)
            } else if let Some(u) = n.as_u64() {
                query.bind(u as i64)
            } else if let Some(f) = n.as_f64() {
                query.bind(f)
            } else {
                query.bind(n.to_string())
            }
        }
        Value::Bool(b) => query.bind(*b),
        Value::Null => query.bind::<Option<i32>>(None),
        other => query.bind(other.to_string()),
    }
}

fn metadata_from_row<R: BackendRow>(row: &R) -> Value {
    // Try read as string first
    if let Ok(Some(s)) = row.get_string("metadata") {
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            return v;
        }
    }
    // Fallback: try bytes (for JSON-typed columns)
    if let Ok(Some(bytes)) = row.get_bytes("metadata") {
        if let Ok(s) = String::from_utf8(bytes) {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                return v;
            }
        }
    }
    Value::Null
}

fn merge_values(
    existing_doc: Option<String>,
    existing_meta: Option<Value>,
    existing_emb: Option<Embedding>,
    new_doc: Option<String>,
    new_meta: Option<Metadata>,
    new_emb: Option<Embedding>,
) -> (Option<String>, Value, Option<Embedding>) {
    let doc = new_doc.or(existing_doc);
    let meta = match (new_meta, existing_meta) {
        (Some(m), _) => m,
        (None, Some(e)) => e,
        (None, None) => Value::Null,
    };
    let emb = new_emb.or(existing_emb);
    (doc, meta, emb)
}

fn empty_query_result(include: Option<&[IncludeField]>) -> QueryResult {
    QueryResult {
        ids: vec![Vec::new()],
        documents: if include_documents(include) {
            Some(vec![Vec::new()])
        } else {
            None
        },
        metadatas: if include_metadatas(include) {
            Some(vec![Vec::new()])
        } else {
            None
        },
        embeddings: if include_embeddings(include) {
            Some(vec![Vec::new()])
        } else {
            None
        },
        distances: Some(vec![Vec::new()]),
    }
}

fn transform_hybrid_rows<R: BackendRow>(rows: Vec<R>, include: Option<&[IncludeField]>) -> QueryResult {
    let mut ids = Vec::new();
    let mut docs = Vec::new();
    let mut metas = Vec::new();
    let mut embs = Vec::new();
    let mut dists = Vec::new();

    for row in rows {
        ids.push(id_from_row(&row));
        if include_documents(include) {
            let doc = row
                .get_string("document")
                .unwrap_or(None)
                .unwrap_or_default();
            docs.push(doc);
        }
        if include_metadatas(include) {
            metas.push(metadata_from_row(&row));
        }
        if include_embeddings(include) {
            let emb = row
                .get_string("embedding")
                .unwrap_or(None)
                .or_else(|| row.get_string("_embedding").unwrap_or(None))
                .map(parse_vector_string)
                .unwrap_or_default();
            embs.push(emb);
        }
        let dist = row
            .get_f32("distance")
            .unwrap_or(None)
            .or_else(|| row.get_f32("_distance").unwrap_or(None))
            .or_else(|| row.get_f32("_score").unwrap_or(None))
            .or_else(|| row.get_f32("score").unwrap_or(None))
            .unwrap_or(0.0);
        dists.push(dist);
    }

    QueryResult {
        ids: vec![ids],
        documents: if include_documents(include) {
            Some(vec![docs])
        } else {
            None
        },
        metadatas: if include_metadatas(include) {
            Some(vec![metas])
        } else {
            None
        },
        embeddings: if include_embeddings(include) {
            Some(vec![embs])
        } else {
            None
        },
        distances: Some(vec![dists]),
    }
}

fn is_hybrid_invalid_argument(err: &SeekDbError) -> bool {
    match err {
        SeekDbError::Sql(msg) => {
            let lower = msg.to_lowercase();
            lower.contains("invalid argument") || lower.contains("1210")
        }
        _ => false,
    }
}

#[derive(serde::Serialize)]
struct HybridSearchParam {
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    knn: Option<HybridKnnExpr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rank: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u32>,
}

#[derive(serde::Serialize)]
struct HybridKnnExpr {
    field: String,
    k: u32,
    query_vector: Embedding,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<Vec<Value>>,
}

async fn build_search_parm_from_typed<Ef: EmbeddingFunction + 'static>(
    collection: &Collection<Ef>,
    query: Option<&HybridQuery>,
    knn: Option<&HybridKnn>,
    rank: Option<&HybridRank>,
    n_results: u32,
) -> Result<String> {
    let query_expr = query.and_then(build_query_expr_from_hybrid);

    let knn_expr = if let Some(knn_cfg) = knn {
        build_knn_expr_from_hybrid(collection, knn_cfg).await?
    } else {
        None
    };

    let rank_value = rank.map(hybrid_rank_to_value);

    if query_expr.is_none() && knn_expr.is_none() && rank_value.is_none() {
        return Ok(String::new());
    }

    let search_parm = HybridSearchParam {
        query: query_expr,
        knn: knn_expr,
        rank: rank_value,
        size: Some(n_results),
    };

    serde_json::to_string(&search_parm).map_err(SeekDbError::Serialization)
}

fn build_query_expr_from_hybrid(query: &HybridQuery) -> Option<Value> {
    let where_doc = query.where_doc.as_ref();
    let where_meta = query.where_meta.as_ref();

    // Case 1: scalar/metadata-only query
    if where_doc.is_none() {
        if let Some(meta) = where_meta {
            let filter_conditions = build_metadata_filter_for_search_parm(meta);
            if filter_conditions.is_empty() {
                return None;
            }
            if filter_conditions.len() == 1 {
                let cond = &filter_conditions[0];
                if cond.get("range").is_some() {
                    return Some(json!({ "range": cond["range"].clone() }));
                } else if cond.get("term").is_some() {
                    return Some(json!({ "term": cond["term"].clone() }));
                } else {
                    return Some(json!({ "bool": { "filter": filter_conditions } }));
                }
            } else {
                return Some(json!({ "bool": { "filter": filter_conditions } }));
            }
        }
        return None;
    }

    // Case 2: full-text query with optional metadata filter
    if let Some(doc_filter) = where_doc {
        let doc_query = build_document_query_for_search_parm(Some(doc_filter));
        if let Some(doc_q) = doc_query {
            let filter_conditions = if let Some(meta) = where_meta {
                build_metadata_filter_for_search_parm(meta)
            } else {
                Vec::new()
            };

            if !filter_conditions.is_empty() {
                return Some(json!({
                    "bool": {
                        "must": [doc_q],
                        "filter": filter_conditions
                    }
                }));
            } else {
                return Some(doc_q);
            }
        }
    }

    None
}

async fn build_knn_expr_from_hybrid<Ef: EmbeddingFunction + 'static>(
    collection: &Collection<Ef>,
    knn: &HybridKnn,
) -> Result<Option<HybridKnnExpr>> {
    if let Some(embs) = &knn.query_embeddings {
        if embs.is_empty() {
            return Err(SeekDbError::InvalidInput(
                "knn.query_embeddings must not be empty".into(),
            ));
        }
        let query_vector = embs[0].clone();
        if query_vector.len() as u32 != collection.dimension {
            return Err(SeekDbError::InvalidInput(format!(
                "embedding dimension {} does not match collection dimension {}",
                query_vector.len(),
                collection.dimension
            )));
        }

        let k = knn.n_results.unwrap_or(10);

        let filter_conditions = if let Some(meta) = &knn.where_meta {
            build_metadata_filter_for_search_parm(meta)
        } else {
            Vec::new()
        };

        let filter = if filter_conditions.is_empty() {
            None
        } else {
            Some(filter_conditions)
        };

        return Ok(Some(HybridKnnExpr {
            field: "embedding".into(),
            k,
            query_vector,
            filter,
        }));
    }

    let Some(texts) = &knn.query_texts else {
        return Err(SeekDbError::InvalidInput(
            "knn requires either query_embeddings or query_texts".into(),
        ));
    };

    if texts.is_empty() {
        return Err(SeekDbError::InvalidInput(
            "knn.query_texts must not be empty".into(),
        ));
    }

    let ef = collection.embedding_function.as_ref().ok_or_else(|| {
        SeekDbError::Embedding(
            "knn.query_texts provided but collection has no embedding_function; provide query_embeddings or set embedding_function."
                .into(),
        )
    })?;

    let first = texts[0].clone();
    let embs = ef.embed_documents(&[first]).await?;
    let Some(query_vector) = embs.into_iter().next() else {
        return Err(SeekDbError::InvalidInput(
            "embedding_function returned empty embeddings for knn.query_texts".into(),
        ));
    };

    if query_vector.len() as u32 != collection.dimension {
        return Err(SeekDbError::InvalidInput(format!(
            "embedding dimension {} does not match collection dimension {}",
            query_vector.len(),
            collection.dimension
        )));
    }

    let k = knn.n_results.unwrap_or(10);

    let filter_conditions = if let Some(meta) = &knn.where_meta {
        build_metadata_filter_for_search_parm(meta)
    } else {
        Vec::new()
    };

    let filter = if filter_conditions.is_empty() {
        None
    } else {
        Some(filter_conditions)
    };

    Ok(Some(HybridKnnExpr {
        field: "embedding".into(),
        k,
        query_vector,
        filter,
    }))
}

fn hybrid_rank_to_value(rank: &HybridRank) -> Value {
    match rank {
        HybridRank::Rrf {
            rank_window_size,
            rank_constant,
        } => {
            let mut inner = serde_json::Map::new();
            if let Some(w) = rank_window_size {
                inner.insert("rank_window_size".to_string(), json!(w));
            }
            if let Some(c) = rank_constant {
                inner.insert("rank_constant".to_string(), json!(c));
            }
            let mut outer = serde_json::Map::new();
            outer.insert("rrf".to_string(), Value::Object(inner));
            Value::Object(outer)
        }
        HybridRank::Raw(v) => v.clone(),
    }
}

fn combine_meta_filters(a: Option<&Filter>, b: Option<&Filter>) -> Option<Filter> {
    match (a, b) {
        (None, None) => None,
        (Some(f), None) | (None, Some(f)) => Some(f.clone()),
        (Some(f1), Some(f2)) => Some(Filter::And(vec![f1.clone(), f2.clone()])),
    }
}

async fn build_search_parm_json<Ef: EmbeddingFunction + 'static>(
    collection: &Collection<Ef>,
    queries: &[String],
    where_meta: Option<&Filter>,
    where_doc: Option<&DocFilter>,
    n_results: u32,
) -> Result<String> {
    let meta_filters = where_meta
        .map(build_metadata_filter_for_search_parm)
        .unwrap_or_default();
    let doc_query = build_document_query_for_search_parm(where_doc);

    let query_expr = if doc_query.is_none() && meta_filters.is_empty() {
        None
    } else if let Some(doc_q) = doc_query {
        if meta_filters.is_empty() {
            Some(doc_q)
        } else {
            Some(json!({ "bool": { "must": [doc_q], "filter": meta_filters } }))
        }
    } else {
        Some(json!({ "bool": { "filter": meta_filters } }))
    };

    let mut knn_expr: Option<HybridKnnExpr> = None;
    if !queries.is_empty() {
        let ef = collection.embedding_function.as_ref().ok_or_else(|| {
            SeekDbError::Embedding(
                "Hybrid search requires embedding_function for text queries; provide search_params with knn.query_vector or set embedding_function."
                    .into(),
            )
        })?;
        let embs = ef.embed_documents(&[queries[0].clone()]).await?;
        let Some(first) = embs.first() else {
            return Err(SeekDbError::InvalidInput(
                "embedding_function returned empty embeddings".into(),
            ));
        };
        if first.len() as u32 != collection.dimension {
            return Err(SeekDbError::InvalidInput(format!(
                "embedding dimension {} does not match collection dimension {}",
                first.len(),
                collection.dimension
            )));
        }
        let knn_filter = if meta_filters.is_empty() {
            None
        } else {
            Some(meta_filters.clone())
        };
        knn_expr = Some(HybridKnnExpr {
            field: "embedding".into(),
            k: n_results,
            query_vector: first.clone(),
            filter: knn_filter,
        });
    }

    if query_expr.is_none() && knn_expr.is_none() {
        return Ok(String::new());
    }

    let search_parm = HybridSearchParam {
        query: query_expr,
        knn: knn_expr,
        rank: None,
        size: Some(n_results),
    };

    serde_json::to_string(&search_parm).map_err(SeekDbError::Serialization)
}

fn build_metadata_filter_for_search_parm(filter: &Filter) -> Vec<Value> {
    match filter {
        Filter::Eq { field, value } => vec![json!({"term": { meta_path(field): value }})],
        Filter::Ne { field, value } => vec![json!({"bool": {"must_not": [ {"term": { meta_path(field): value }} ]}})],
        Filter::Gt { field, value } => vec![json!({"range": { meta_path(field): { "gt": value }}})],
        Filter::Gte { field, value } => vec![json!({"range": { meta_path(field): { "gte": value }}})],
        Filter::Lt { field, value } => vec![json!({"range": { meta_path(field): { "lt": value }}})],
        Filter::Lte { field, value } => vec![json!({"range": { meta_path(field): { "lte": value }}})],
        Filter::In { field, values } => vec![json!({"terms": { meta_path(field): values }})],
        Filter::Nin { field, values } => vec![json!({"bool": { "must_not": [ {"terms": { meta_path(field): values }} ]}})],
        Filter::And(filters) => {
            let mut parts = Vec::new();
            for f in filters {
                let sub = build_metadata_filter_for_search_parm(f);
                if sub.len() == 1 {
                    parts.push(sub[0].clone());
                } else if !sub.is_empty() {
                    parts.push(json!({"bool": {"must": sub}}));
                }
            }
            if parts.is_empty() {
                Vec::new()
            } else {
                vec![json!({"bool": {"must": parts}})]
            }
        }
        Filter::Or(filters) => {
            let mut parts = Vec::new();
            for f in filters {
                let sub = build_metadata_filter_for_search_parm(f);
                if sub.len() == 1 {
                    parts.push(sub[0].clone());
                } else if !sub.is_empty() {
                    parts.push(json!({"bool": {"must": sub}}));
                }
            }
            if parts.is_empty() {
                Vec::new()
            } else {
                vec![json!({"bool": {"should": parts}})]
            }
        }
        Filter::Not(sub) => {
            let sub_filters = build_metadata_filter_for_search_parm(sub);
            if sub_filters.is_empty() {
                Vec::new()
            } else {
                vec![json!({"bool": { "must_not": sub_filters }})]
            }
        }
    }
}

fn meta_path(field: &str) -> String {
    format!("(JSON_EXTRACT(metadata, '$.{field}'))")
}

fn build_document_query_for_search_parm(where_doc: Option<&DocFilter>) -> Option<Value> {
    let Some(filter) = where_doc else { return None };
    match filter {
        DocFilter::Contains(text) => Some(json!({"query_string": { "fields": ["document"], "query": text } })),
        DocFilter::And(filters) => {
            let mut parts = Vec::new();
            for f in filters {
                if let DocFilter::Contains(text) = f {
                    parts.push(text.clone());
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(json!({"query_string": { "fields": ["document"], "query": parts.join(" ") } }))
            }
        }
        DocFilter::Or(filters) => {
            let mut parts = Vec::new();
            for f in filters {
                if let DocFilter::Contains(text) = f {
                    parts.push(text.clone());
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(json!({"query_string": { "fields": ["document"], "query": parts.join(" OR ") } }))
            }
        }
        DocFilter::Regex(_) => None, // not supported in hybrid search parameter builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_vector_roundtrip() {
        let v = vec![1.0, 2.5, 3.0];
        let s = vector_to_string(&v);
        assert_eq!(s, "[1,2.5,3]");
        assert_eq!(parse_vector_string(s), v);
    }

    #[test]
    fn test_validate_lengths_dimension_mismatch() {
        let ids = vec!["a".into()];
        let embeddings = vec![vec![0.1_f32, 0.2_f32]];
        let err = validate_lengths(&ids, &embeddings, None, None, 3).unwrap_err();
        assert!(matches!(err, SeekDbError::InvalidInput(_)));
    }

    #[test]
    fn test_merge_values() {
        let (doc, meta, emb) = merge_values(
            Some("old".into()),
            Some(json!({"x":1})),
            Some(vec![1.0]),
            None,
            Some(json!({"x":2})),
            None,
        );
        assert_eq!(doc.unwrap(), "old");
        assert_eq!(meta["x"], 2);
        assert!(emb.is_some());
    }
}
