//! Model manager for lazy loading embedder and reranker models.
//!
//! This module provides lazy-loaded access to embedding and reranking models,
//! supporting graceful fallback when models are unavailable.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use frankensearch::core::filter::SearchFilter as FsSearchFilter;
use frankensearch::index::{
    HNSW_DEFAULT_EF_SEARCH as FS_HNSW_DEFAULT_EF_SEARCH, HnswIndex as FsHnswIndex,
    VectorIndex as FsVectorIndex,
};
use parking_lot::RwLock;
use tracing::{info, warn};

use crate::search::ann_index::AnnSearchStats;
use crate::search::embedder::{Embedder, EmbedderError, EmbedderResult};
use crate::search::fastembed_embedder::FastEmbedder;
use crate::search::fastembed_reranker::FastEmbedReranker;
use crate::search::hash_embedder::HashEmbedder;
use crate::search::reranker::{Reranker, RerankerError, RerankerResult, rerank_texts};
use crate::search::vector_index::{
    SemanticFilter, VectorSearchResult, parse_semantic_doc_id, vector_index_path,
};
use crate::search::ann_index::hnsw_index_path;

const ANN_CANDIDATE_MULTIPLIER: usize = 4;

struct SemanticIndexPair {
    vector_index: Arc<FsVectorIndex>,
    ann_index: Arc<FsHnswIndex>,
}

/// Model manager that handles lazy loading of embedder and reranker models.
pub struct ModelManager {
    data_dir: PathBuf,
    embedder: RwLock<Option<Arc<dyn Embedder>>>,
    reranker: RwLock<Option<Arc<dyn Reranker>>>,
    embedder_name: RwLock<String>,
    reranker_name: RwLock<String>,
    fallback_embedder: Arc<HashEmbedder>,
    semantic_indices: RwLock<HashMap<(PathBuf, PathBuf), Arc<SemanticIndexPair>>>,
}

impl ModelManager {
    /// Create a new model manager with the given data directory.
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            embedder: RwLock::new(None),
            reranker: RwLock::new(None),
            embedder_name: RwLock::new("not-loaded".to_string()),
            reranker_name: RwLock::new("not-loaded".to_string()),
            fallback_embedder: Arc::new(HashEmbedder::new(384)),
            semantic_indices: RwLock::new(HashMap::new()),
        }
    }

    /// Check if any model is loaded and ready.
    pub fn is_ready(&self) -> bool {
        self.embedder.read().is_some()
    }

    /// Get the embedder ID.
    pub fn embedder_id(&self) -> String {
        self.embedder
            .read()
            .as_ref()
            .map(|e| e.id().to_string())
            .unwrap_or_else(|| "hash-384".to_string())
    }

    /// Get the embedder name.
    pub fn embedder_name(&self) -> String {
        self.embedder_name.read().clone()
    }

    /// Get the embedder dimension.
    pub fn embedder_dimension(&self) -> usize {
        self.embedder
            .read()
            .as_ref()
            .map(|e| e.dimension())
            .unwrap_or(384)
    }

    /// Check if embedder is loaded.
    pub fn embedder_loaded(&self) -> bool {
        self.embedder.read().is_some()
    }

    /// Get the reranker ID.
    pub fn reranker_id(&self) -> String {
        self.reranker
            .read()
            .as_ref()
            .map(|r| r.id().to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    /// Get the reranker name.
    pub fn reranker_name(&self) -> String {
        self.reranker_name.read().clone()
    }

    /// Check if reranker is loaded.
    pub fn reranker_loaded(&self) -> bool {
        self.reranker.read().is_some()
    }

    /// Pre-warm the embedder by loading it.
    pub fn warm_embedder(&self) -> EmbedderResult<()> {
        // Fast path: already loaded
        if self.embedder.read().is_some() {
            return Ok(());
        }

        // Slow path: need to load. Take write lock and check again.
        let mut embedder_guard = self.embedder.write();
        if embedder_guard.is_some() {
            return Ok(());
        }

        let model_dir = FastEmbedder::default_model_dir(&self.data_dir);
        info!(model_dir = %model_dir.display(), "Loading embedder");

        match FastEmbedder::load_from_dir(&model_dir) {
            Ok(embedder) => {
                let id = embedder.id().to_string();
                let dimension = embedder.dimension();
                *embedder_guard = Some(Arc::new(embedder));
                *self.embedder_name.write() = "MiniLM-L6-v2".to_string();
                info!(id = %id, dimension = dimension, "Embedder loaded");
                Ok(())
            }
            Err(e) => {
                warn!(error = %e, "Failed to load embedder, using hash fallback");
                *embedder_guard = Some(self.fallback_embedder.clone());
                *self.embedder_name.write() = "hash-fallback".to_string();
                // Return Ok since we have a fallback
                Ok(())
            }
        }
    }

    /// Pre-warm the reranker by loading it.
    pub fn warm_reranker(&self) -> RerankerResult<()> {
        // Fast path: already loaded
        if self.reranker.read().is_some() {
            return Ok(());
        }

        // Slow path: need to load. Take write lock and check again.
        let mut reranker_guard = self.reranker.write();
        if reranker_guard.is_some() {
            return Ok(());
        }

        let model_dir = FastEmbedReranker::default_model_dir(&self.data_dir);
        info!(model_dir = %model_dir.display(), "Loading reranker");

        match FastEmbedReranker::load_from_dir(&model_dir) {
            Ok(reranker) => {
                let id = reranker.id().to_string();
                *reranker_guard = Some(Arc::new(reranker));
                *self.reranker_name.write() = "ms-marco-MiniLM-L-6-v2".to_string();
                info!(id = %id, "Reranker loaded");
                Ok(())
            }
            Err(e) => {
                warn!(error = %e, "Failed to load reranker, reranking unavailable");
                Err(e)
            }
        }
    }

    /// Embed a batch of texts.
    pub fn embed_batch(&self, texts: &[String]) -> EmbedderResult<Vec<Vec<f32>>> {
        // Ensure embedder is loaded
        if self.embedder.read().is_none() {
            self.warm_embedder()?;
        }

        let embedder = self.embedder.read();
        let embedder = embedder
            .as_ref()
            .ok_or_else(|| EmbedderError::EmbedderUnavailable {
                model: "unknown".to_string(),
                reason: "embedder not loaded".to_string(),
            })?;

        // Convert to &str slice for the batch call
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        embedder.embed_batch_sync(&text_refs)
    }

    /// Embed a single text.
    pub fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        // Ensure embedder is loaded
        if self.embedder.read().is_none() {
            self.warm_embedder()?;
        }

        let embedder = self.embedder.read();
        let embedder = embedder
            .as_ref()
            .ok_or_else(|| EmbedderError::EmbedderUnavailable {
                model: "unknown".to_string(),
                reason: "embedder not loaded".to_string(),
            })?;

        embedder.embed_sync(text)
    }

    /// Rerank documents against a query.
    pub fn rerank(&self, query: &str, documents: &[String]) -> RerankerResult<Vec<f32>> {
        // Ensure reranker is loaded
        if self.reranker.read().is_none() {
            self.warm_reranker()?;
        }

        let reranker = self.reranker.read();
        let reranker = reranker
            .as_ref()
            .ok_or_else(|| RerankerError::RerankerUnavailable {
                model: "reranker".to_string(),
            })?;

        // Convert to &str slice and use rerank_texts bridge
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        rerank_texts(&**reranker, query, &doc_refs)
    }

    fn cached_semantic_pair(
        &self,
        vector_index_path: &Path,
        ann_path: &Path,
    ) -> Result<Arc<SemanticIndexPair>> {
        let key = (vector_index_path.to_path_buf(), ann_path.to_path_buf());
        let mut guard = self.semantic_indices.write();
        if let Some(existing) = guard.get(&key).cloned() {
            return Ok(existing);
        }

        let vector_index = Arc::new(FsVectorIndex::open(vector_index_path).with_context(|| {
            format!(
                "open semantic vector index failed: {}",
                vector_index_path.display()
            )
        })?);
        if !ann_path.is_file() {
            bail!("HNSW index not found at {}", ann_path.display());
        }
        let ann_index = Arc::new(
            FsHnswIndex::load(ann_path, vector_index.as_ref())
                .map_err(|err| anyhow::anyhow!("open HNSW index failed: {err}"))?,
        );
        let matches = ann_index
            .matches_vector_index(vector_index.as_ref())
            .map_err(|err| anyhow::anyhow!("validate HNSW index failed: {err}"))?;
        if !matches {
            bail!(
                "HNSW index at {} is stale for current semantic index",
                ann_path.display()
            );
        }

        let pair = Arc::new(SemanticIndexPair {
            vector_index,
            ann_index,
        });
        guard.insert(key, Arc::clone(&pair));
        Ok(pair)
    }

    pub fn warm_semantic_assets(&self) -> Result<()> {
        let embedder_id = self.embedder_id();
        let vector_path = vector_index_path(&self.data_dir, &embedder_id);
        let ann_path = hnsw_index_path(&self.data_dir, &embedder_id);
        if !vector_path.is_file() || !ann_path.is_file() {
            bail!(
                "semantic assets missing for {} (expected {}, {})",
                embedder_id,
                vector_path.display(),
                ann_path.display()
            );
        }
        let _ = self.cached_semantic_pair(&vector_path, &ann_path)?;
        Ok(())
    }

    pub fn semantic_search_approx(
        &self,
        vector_index_path: &Path,
        ann_path: &Path,
        embedding: &[f32],
        fetch_limit: usize,
        filter: &SemanticFilter,
    ) -> Result<(Vec<VectorSearchResult>, AnnSearchStats)> {
        let pair = self.cached_semantic_pair(vector_index_path, ann_path)?;
        let candidate = fetch_limit
            .saturating_mul(ANN_CANDIDATE_MULTIPLIER)
            .max(fetch_limit);
        let ef = FS_HNSW_DEFAULT_EF_SEARCH.max(candidate);
        let (ann_results, search_stats) = pair
            .ann_index
            .knn_search_with_stats(embedding, candidate, ef)
            .map_err(|err| anyhow::anyhow!("frankensearch approximate search failed: {err}"))?;

        let mut best_by_message: HashMap<u64, VectorSearchResult> = HashMap::new();
        for hit in ann_results {
            if !filter.is_unrestricted() && !filter.matches(&hit.doc_id, None) {
                continue;
            }
            let Some(parsed) = parse_semantic_doc_id(&hit.doc_id) else {
                continue;
            };
            best_by_message
                .entry(parsed.message_id)
                .and_modify(|entry| {
                    if hit.score > entry.score {
                        entry.score = hit.score;
                        entry.chunk_idx = parsed.chunk_idx;
                    }
                })
                .or_insert(VectorSearchResult {
                    message_id: parsed.message_id,
                    chunk_idx: parsed.chunk_idx,
                    score: hit.score,
                });
        }

        let mut results: Vec<VectorSearchResult> = best_by_message.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.message_id.cmp(&b.message_id))
        });
        if results.len() > fetch_limit {
            results.truncate(fetch_limit);
        }

        Ok((
            results,
            AnnSearchStats {
                index_size: search_stats.index_size,
                dimension: search_stats.dimension,
                ef_search: search_stats.ef_search,
                k_requested: search_stats.k_requested,
                k_returned: search_stats.k_returned,
                search_time_us: search_stats.search_time_us,
                estimated_recall: search_stats.estimated_recall as f32,
                is_approximate: search_stats.is_approximate,
            },
        ))
    }

    /// Unload all models to free memory.
    pub fn unload_all(&self) {
        *self.embedder.write() = None;
        *self.reranker.write() = None;
        *self.embedder_name.write() = "not-loaded".to_string();
        *self.reranker_name.write() = "not-loaded".to_string();
        self.semantic_indices.write().clear();
        info!("All models unloaded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    #[allow(dead_code)]
    fn model_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/models/xenova-paraphrase-minilm-l3-v2-int8")
    }

    #[test]
    fn test_model_manager_creation() {
        let manager = ModelManager::new(&test_data_dir());
        assert!(!manager.is_ready());
        assert!(!manager.embedder_loaded());
        assert!(!manager.reranker_loaded());
    }

    #[test]
    fn test_embedder_fallback_on_missing_model() {
        // Use a directory without models
        let manager = ModelManager::new(&PathBuf::from("/tmp/nonexistent"));

        // Should succeed with fallback
        let result = manager.warm_embedder();
        assert!(result.is_ok());

        // Should be using hash fallback
        assert!(manager.embedder_loaded());
        assert_eq!(manager.embedder_name(), "hash-fallback");
    }

    #[test]
    fn test_embedder_dimension() {
        let manager = ModelManager::new(&test_data_dir());
        // Before loading, should return default dimension
        assert_eq!(manager.embedder_dimension(), 384);
    }

    #[test]
    fn test_unload_all() {
        let manager = ModelManager::new(&test_data_dir());
        let _ = manager.warm_embedder();

        assert!(manager.embedder_loaded());

        manager.unload_all();

        assert!(!manager.embedder_loaded());
        assert!(!manager.reranker_loaded());
    }

    #[test]
    fn test_embed_with_fallback() {
        let manager = ModelManager::new(&PathBuf::from("/tmp/nonexistent"));

        // Should work with fallback
        let result = manager.embed("test text");
        assert!(result.is_ok());

        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 384);
    }
}
