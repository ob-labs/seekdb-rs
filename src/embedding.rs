use async_trait::async_trait;

use crate::error::Result;
#[cfg(feature = "embedding")]
use crate::error::SeekDbError;
use crate::types::Embeddings;

/// Embedding generation abstraction to allow custom models.
#[async_trait]
pub trait EmbeddingFunction: Send + Sync {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings>;
    fn dimension(&self) -> usize;
}

/// Convenience impl so that `Box<dyn EmbeddingFunction>` can be used
/// as the generic parameter for `Collection<Ef>`.
#[async_trait]
impl EmbeddingFunction for Box<dyn EmbeddingFunction> {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings> {
        (**self).embed_documents(docs).await
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }
}

/// Default ONNX-based embedding implementation (all-MiniLM-L6-v2).
/// Compiled only when the `embedding` feature is enabled.
#[cfg(feature = "embedding")]
pub struct DefaultEmbedding {
    tokenizer: tokenizers::Tokenizer,
    session: std::sync::Arc<std::sync::Mutex<ort::session::Session>>,
    max_length: usize,
}

#[cfg(feature = "embedding")]
impl DefaultEmbedding {
    pub fn new() -> Result<Self> {
        let (model_path, tokenizer_path) = resolve_model_paths()?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| SeekDbError::Embedding(format!("failed to load tokenizer: {e}")))?;

        // Configure truncation/padding to fixed max_length.
        let mut trunc = tokenizer.get_truncation().cloned().unwrap_or_else(|| {
            tokenizers::utils::truncation::TruncationParams {
                max_length: DEFAULT_MAX_LENGTH,
                ..Default::default()
            }
        });
        trunc.max_length = DEFAULT_MAX_LENGTH;
        tokenizer
            .with_truncation(Some(trunc))
            .map_err(|e| SeekDbError::Embedding(format!("failed to set truncation: {e}")))?;

        let mut padding = tokenizer.get_padding().cloned().unwrap_or_default();
        padding.strategy = tokenizers::utils::padding::PaddingStrategy::Fixed(DEFAULT_MAX_LENGTH);
        tokenizer.with_padding(Some(padding));

        // Build ONNX Runtime session
        let session = ort::session::Session::builder().map_err(|e| {
            SeekDbError::Embedding(format!("failed to create session builder: {e}"))
        })?;
        let session = session
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
            .map_err(|e| SeekDbError::Embedding(format!("failed to set optimization level: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| {
                SeekDbError::Embedding(format!(
                    "failed to load onnx model from {}: {e}",
                    model_path.display()
                ))
            })?;

        Ok(Self {
            tokenizer,
            session: std::sync::Arc::new(std::sync::Mutex::new(session)),
            max_length: DEFAULT_MAX_LENGTH,
        })
    }
}

#[cfg(feature = "embedding")]
#[async_trait]
impl EmbeddingFunction for DefaultEmbedding {
    async fn embed_documents(&self, docs: &[String]) -> Result<Embeddings> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        run_inference(&self.session, &self.tokenizer, docs, self.max_length)
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

#[cfg(feature = "embedding")]
const HF_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
#[cfg(feature = "embedding")]
const DEFAULT_MAX_LENGTH: usize = 512;
#[cfg(feature = "embedding")]
const EMBEDDING_DIM: usize = 384;

#[cfg(feature = "embedding")]
fn cache_root() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("SEEKDB_ONNX_CACHE_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(".cache/seekdb/onnx_models")
}

#[cfg(feature = "embedding")]
fn resolve_model_paths() -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};
    use std::path::PathBuf;

    // Relative paths inside the model repo; can be overridden for non-standard layouts.
    let model_rel =
        std::env::var("SEEKDB_ONNX_MODEL_PATH").unwrap_or_else(|_| "onnx/model.onnx".to_string());
    let tokenizer_rel = std::env::var("SEEKDB_ONNX_TOKENIZER_PATH")
        .unwrap_or_else(|_| "tokenizer.json".to_string());

    // 1) If user explicitly provides a local model directory, use it and skip network.
    if let Ok(dir) = std::env::var("SEEKDB_ONNX_MODEL_DIR") {
        let root = PathBuf::from(dir);
        let model_path = root.join(&model_rel);
        let tokenizer_path = root.join(&tokenizer_rel);

        if !model_path.exists() {
            return Err(SeekDbError::Embedding(format!(
                "model.onnx not found at {} (SEEKDB_ONNX_MODEL_PATH={model_rel})",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(SeekDbError::Embedding(format!(
                "tokenizer.json not found at {} (SEEKDB_ONNX_TOKENIZER_PATH={tokenizer_rel})",
                tokenizer_path.display()
            )));
        }
        return Ok((model_path, tokenizer_path));
    }

    // 2) Otherwise, download / reuse model files via hf-hub into a configurable cache.
    let cache_dir = cache_root();
    let api = ApiBuilder::from_env()
        .with_cache_dir(cache_dir)
        .with_progress(true)
        .build()
        .map_err(|e| SeekDbError::Embedding(format!("failed to create hf-hub Api: {e}")))?;

    let repo_id = std::env::var("SEEKDB_ONNX_REPO_ID").unwrap_or_else(|_| HF_MODEL_ID.to_string());
    let revision = std::env::var("SEEKDB_ONNX_REVISION").unwrap_or_else(|_| "main".to_string());

    let repo = Repo::with_revision(repo_id, RepoType::Model, revision);
    let api_repo = api.repo(repo);

    let model_path = api_repo.get(&model_rel).map_err(|e| {
        SeekDbError::Embedding(format!("failed to get {model_rel} from hf-hub: {e}"))
    })?;
    let tokenizer_path = api_repo.get(&tokenizer_rel).map_err(|e| {
        SeekDbError::Embedding(format!("failed to get {tokenizer_rel} from hf-hub: {e}"))
    })?;

    Ok((model_path, tokenizer_path))
}

#[cfg(feature = "embedding")]
fn run_inference(
    session: &std::sync::Arc<std::sync::Mutex<ort::session::Session>>,
    tokenizer: &tokenizers::Tokenizer,
    docs: &[String],
    max_length: usize,
) -> Result<Embeddings> {
    use tokenizers::utils::{padding::PaddingStrategy, truncation::TruncationParams};

    if docs.is_empty() {
        return Ok(Vec::new());
    }

    let mut tokenizer = tokenizer.clone();
    // Ensure padding/truncation are set (defensive if caller forgot).
    let mut trunc = tokenizer
        .get_truncation()
        .cloned()
        .unwrap_or_else(|| TruncationParams {
            max_length,
            ..Default::default()
        });
    trunc.max_length = max_length;
    tokenizer
        .with_truncation(Some(trunc))
        .map_err(|e| SeekDbError::Embedding(format!("failed to set truncation: {e}")))?;

    let mut padding = tokenizer.get_padding().cloned().unwrap_or_default();
    padding.strategy = PaddingStrategy::Fixed(max_length);
    tokenizer.with_padding(Some(padding));

    let encodings = tokenizer
        .encode_batch(docs.to_vec(), true)
        .map_err(|e| SeekDbError::Embedding(format!("tokenization failed: {e}")))?;

    let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);
    if seq_len == 0 {
        return Err(SeekDbError::Embedding(
            "tokenization produced empty sequence".into(),
        ));
    }

    let batch = encodings.len();
    let mut input_ids: Vec<i64> = Vec::with_capacity(batch * seq_len);
    let mut attention_mask: Vec<i64> = Vec::with_capacity(batch * seq_len);
    let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch * seq_len);
    for enc in &encodings {
        if enc.get_ids().len() != seq_len || enc.get_attention_mask().len() != seq_len {
            return Err(SeekDbError::Embedding(
                "tokenization produced inconsistent sequence lengths".into(),
            ));
        }
        input_ids.extend(enc.get_ids().iter().map(|id| *id as i64));
        attention_mask.extend(enc.get_attention_mask().iter().map(|m| *m as i64));
        token_type_ids.extend(std::iter::repeat(0_i64).take(seq_len));
    }

    let shape: Vec<i64> = vec![batch as i64, seq_len as i64];
    let input_ids_tensor = ort::value::Tensor::<i64>::from_array((shape.clone(), input_ids))
        .map_err(|e| SeekDbError::Embedding(format!("failed to build input_ids tensor: {e}")))?;
    let attention_tensor =
        ort::value::Tensor::<i64>::from_array((shape.clone(), attention_mask.clone())).map_err(
            |e| SeekDbError::Embedding(format!("failed to build attention_mask tensor: {e}")),
        )?;
    let token_type_tensor = ort::value::Tensor::<i64>::from_array((shape.clone(), token_type_ids))
        .map_err(|e| {
            SeekDbError::Embedding(format!("failed to build token_type_ids tensor: {e}"))
        })?;

    let mut session_guard = session
        .lock()
        .map_err(|_| SeekDbError::Embedding("failed to lock onnx session".into()))?;
    let outputs = session_guard
        .run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_tensor,
            "token_type_ids" => token_type_tensor
        ])
        .map_err(|e| SeekDbError::Embedding(format!("onnx run failed: {e}")))?;

    if outputs.len() == 0 {
        return Err(SeekDbError::Embedding(
            "onnx model returned no outputs".into(),
        ));
    }
    let output = &outputs[0];
    let (out_shape, out_data) = output
        .try_extract_tensor::<f32>()
        .map_err(|e| SeekDbError::Embedding(format!("failed to extract tensor: {e}")))?;

    if out_shape.len() != 3 {
        return Err(SeekDbError::Embedding(format!(
            "unexpected output shape: {out_shape:?}"
        )));
    }
    let out_batch = out_shape[0] as usize;
    let out_seq_len = out_shape[1] as usize;
    let hidden = out_shape[2] as usize;

    if out_batch != batch || out_seq_len != seq_len || hidden != EMBEDDING_DIM {
        return Err(SeekDbError::Embedding(format!(
            "unexpected output dims (got {out_batch}x{out_seq_len}x{hidden}, expected {batch}x{seq_len}x{EMBEDDING_DIM})"
        )));
    }

    mean_pool(out_data, &attention_mask, batch, seq_len, hidden)
}

#[cfg(feature = "embedding")]
fn mean_pool(
    data: &[f32],
    attention_mask: &[i64],
    batch: usize,
    seq_len: usize,
    hidden: usize,
) -> Result<Embeddings> {
    if attention_mask.len() != batch * seq_len {
        return Err(SeekDbError::Embedding(
            "attention mask length does not match batch and sequence length".into(),
        ));
    }
    if data.len() != batch * seq_len * hidden {
        return Err(SeekDbError::Embedding(
            "model output size does not match expected dimensions".into(),
        ));
    }

    let mut outputs = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut vec = vec![0f32; hidden];
        let mut count = 0f32;
        for t in 0..seq_len {
            if attention_mask[b * seq_len + t] == 0 {
                continue;
            }
            count += 1.0;
            let offset = (b * seq_len + t) * hidden;
            for h in 0..hidden {
                vec[h] += data[offset + h];
            }
        }
        if count == 0.0 {
            count = 1.0; // avoid div0, though attention_mask should have at least CLS token.
        }
        for v in vec.iter_mut() {
            *v /= count;
        }
        outputs.push(vec);
    }
    Ok(outputs)
}

#[cfg(all(test, feature = "embedding"))]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pool_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
        let mask = vec![1, 1];
        let pooled = mean_pool(&data, &mask, 1, 2, 3).unwrap();
        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0], vec![2.5, 4.0, 5.5]);
    }

    #[test]
    fn test_mean_pool_ignores_masked() {
        let data = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let mask = vec![1, 0];
        let pooled = mean_pool(&data, &mask, 1, 2, 3).unwrap();
        assert_eq!(pooled[0], vec![1.0, 1.0, 1.0]);
    }

    /// Basic smoke test for DefaultEmbedding end-to-end ONNX inference.
    #[test]
    fn default_embedding_infers_shape() {
        let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(async {
            let ef = DefaultEmbedding::new().expect("failed to create DefaultEmbedding");
            let docs = vec!["hello world".to_string(), "seekdb rust".to_string()];
            let embs = ef
                .embed_documents(&docs)
                .await
                .expect("embed_documents failed");
            assert_eq!(embs.len(), 2);
            assert_eq!(embs[0].len(), EMBEDDING_DIM);
            assert_eq!(embs[1].len(), EMBEDDING_DIM);
        });
    }
}
