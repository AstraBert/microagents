use ignore::WalkBuilder;
use indicatif::ProgressIterator;
use liteparse::{LiteParse, LiteParseConfig};
use microagents_core::agent::SupportedProvider;
use microagents_core::common::tokenizer;
use microagents_core::types::AgentError;
use qdrant_edge::{
    AnyVariants, Condition, Distance, EdgeConfig, EdgeShard, EdgeSparseVectorParams,
    EdgeVectorParams, FieldCondition, Filter, HnswIndexConfig, JsonPath, Match, PointId,
    PointInsertOperations, PointOperations, PointStruct, PointStructPersisted, QuantizationConfig,
    SparseVector, UpdateOperation, Vector, Vectors,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use tokio::sync::{Mutex, Semaphore, mpsc};
use tokio::task::JoinSet;

use crate::processing::{chunk, embed};

const FILES_INDEX: &str = ".microagents/files_index.json";
const SHARD_DIRECTORY: &str = ".microagents/vectors";
pub const VECTORS_NAME: &str = "text";
pub const CODE_VECTORS_NAME: &str = "code";
pub const SPARSE_VECTORS_NAME: &str = "sparse";
const VECTORS_SIZE: usize = 768;
const CODE_VECTORS_SIZE: usize = 768;
const INGESTION_CONCURRENCY: usize = 10;
// OCR is disabled, images are not supported
pub const SUPPORTED_LIT_EXTENSIONS: &[&str] = &[
    ".pdf", ".doc", ".docx", ".docm", ".odt", ".rtf", ".ppt", ".pptx", ".pptm", ".odp", ".xls",
    ".xlsx", ".xlsm", ".ods", ".csv", ".tsv",
];
pub const SUPPORT_ENV_VARIABLES: &[(&str, &str, &str)] = &[
    ("OPENROUTER_API_KEY", "", "openrouter"),
    ("OPENAI_API_KEY", "OPENAI_BASE_URL", "openai-compatible"),
    ("OPENAI_API_KEY", "", "openai"),
    ("GROQ_API_KEY", "", "groq"),
    ("", "OLLAMA_BASE_URL", "ollama"),
];
static EDGE_CONFIG: OnceLock<EdgeConfig> = OnceLock::new();
pub static PARSER_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

pub fn edge_config() -> &'static EdgeConfig {
    EDGE_CONFIG.get_or_init(|| {
        let config: QuantizationConfig = serde_json::from_value(serde_json::json!({
            "turbo": {
                "always_ram": true,
                "bits": "bits4"
            }
        }))
        .expect("Should be able to generate quantization config");
        EdgeConfig {
            on_disk_payload: true,
            vectors: HashMap::from([
                (
                    VECTORS_NAME.to_string(),
                    EdgeVectorParams {
                        size: VECTORS_SIZE,
                        distance: Distance::Cosine,
                        on_disk: Some(true),
                        quantization_config: Some(config.clone()),
                        multivector_config: None,
                        datatype: None,
                        hnsw_config: None,
                    },
                ),
                (
                    CODE_VECTORS_NAME.to_string(),
                    EdgeVectorParams {
                        size: CODE_VECTORS_SIZE,
                        distance: Distance::Cosine,
                        on_disk: Some(true),
                        quantization_config: Some(config),
                        multivector_config: None,
                        datatype: None,
                        hnsw_config: None,
                    },
                ),
            ]),
            sparse_vectors: HashMap::from([(
                SPARSE_VECTORS_NAME.to_string(),
                EdgeSparseVectorParams {
                    modifier: Some(qdrant_edge::Modifier::Idf),
                    ..Default::default()
                },
            )]),
            hnsw_config: HnswIndexConfig {
                m: 60,
                ef_construct: 800,
                ..Default::default()
            },
            quantization_config: None,
            optimizers: Default::default(),
            wal_options: None,
        }
    })
}

pub fn parser() -> LiteParse {
    #[allow(clippy::needless_update)]
    LiteParse::new(LiteParseConfig {
        ocr_enabled: false,
        ocr_language: "eng".into(),
        ocr_server_url: None,
        tessdata_path: None,
        max_pages: 500,
        password: None,
        target_pages: None,
        dpi: 120_f32,
        output_format: liteparse::OutputFormat::Text,
        preserve_very_small_text: false,
        quiet: true,
        num_workers: 1,
        ..Default::default()
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Document {
    path: String,
    fingerprint: String,
}

impl Document {
    fn has_been_modified(&self, other: &Document) -> bool {
        other.fingerprint != self.fingerprint
    }
}

struct Diff {
    modified: HashSet<String>,
    deleted: HashSet<String>,
    created: HashSet<String>,
}

impl Diff {
    fn is_no_diff(&self) -> bool {
        self.created.is_empty() && self.deleted.is_empty() && self.modified.is_empty()
    }

    fn to_reingest(&self) -> HashSet<String> {
        let mut combined = self.created.clone();
        combined.extend(self.modified.iter().cloned());
        combined
    }

    fn to_delete(&self) -> HashSet<String> {
        let mut combined = self.deleted.clone();
        combined.extend(self.modified.iter().cloned());
        combined
    }
}

impl Document {
    fn new(path: String, size: u64, mtime: u128) -> Self {
        Self {
            path,
            fingerprint: format!("{:?}-{:?}", size, mtime),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingPayload {
    pub document_path: String,
    pub content: String,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
}

#[derive(Debug, Clone)]
struct EmbeddingWithPayload {
    embedding: Vec<f32>,
    code_embedding: Vec<f32>,
    sparse_embedding: SparseVector,
    payload: EmbeddingPayload,
}

/// Walk up from CWD looking for the files index. Returns `Ok(Some(root))` if
/// found, `Ok(None)` if no ancestor contains it, or `Err` on I/O errors.
fn resolve_root_path() -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let current_dir = std::env::current_dir()?;
    for ancestor in current_dir.ancestors() {
        if ancestor.join(FILES_INDEX).exists() {
            return Ok(Some(ancestor.to_path_buf()));
        }
    }
    Ok(None)
}

/// Resolve the project root, falling back to CWD if no index file exists yet.
/// Always returns an absolute path so downstream operations (file walking,
/// reading) are independent of the process CWD.
pub fn root_or_cwd() -> Result<PathBuf, Box<dyn std::error::Error>> {
    match resolve_root_path()? {
        Some(p) => Ok(p),
        None => Ok(std::env::current_dir()?),
    }
}

fn collect_files() -> Result<HashMap<String, Document>, Box<dyn std::error::Error>> {
    let mut paths: HashMap<String, Document> = HashMap::new();
    let root_path = root_or_cwd()?;
    let walker = WalkBuilder::new(&root_path)
        .hidden(false) // include dotfiles if needed
        .add_custom_ignore_filename(".microagentsignore")
        .build();
    for entry in walker {
        let entry = entry?;
        let path = entry.into_path();
        let meta = fs::metadata(&path)?;
        if path.is_file() {
            let repl = path.to_string_lossy().replace('\\', "/");
            paths.insert(
                repl.clone(),
                Document::new(
                    repl,
                    meta.len(),
                    meta.modified()?
                        .duration_since(SystemTime::UNIX_EPOCH)?
                        .as_millis(),
                ),
            );
        }
    }
    Ok(paths)
}

fn persist_file_changes(new_content: String) -> Result<(), Box<dyn std::error::Error>> {
    let root_path = root_or_cwd()?;
    let p = root_path.join(FILES_INDEX);
    if p.exists() {
        let mut tmp_path = tempfile::NamedTempFile::new_in(&root_path)?;
        tmp_path.write_all(new_content.as_bytes())?;
        tmp_path.flush()?;
        tmp_path.persist(&p)?;
        return Ok(());
    }
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&p, new_content)?;
    Ok(())
}

fn diff_files(files: HashMap<String, Document>) -> Result<Diff, Box<dyn std::error::Error>> {
    let root_path = root_or_cwd()?;
    let p = root_path.join(FILES_INDEX);
    if p.exists() {
        let content = fs::read_to_string(&p)?;
        let fls: HashMap<String, Document> = serde_json::from_str(&content)?;
        let paths_now: HashSet<&str> = files.keys().map(|k| k.as_str()).collect();
        let paths_before: HashSet<&str> = fls.keys().map(|k| k.as_str()).collect();

        let created: HashSet<&str> = paths_now.difference(&paths_before).copied().collect();
        let deleted: HashSet<&str> = paths_before.difference(&paths_now).copied().collect();
        let mut modified: HashSet<&str> = HashSet::new();
        for (k, v) in fls.clone() {
            if !created.contains(k.as_str()) && !deleted.contains(k.as_str()) {
                let fl = files.get(&k).unwrap(); // we know that this is shared
                if fl.has_been_modified(&v) {
                    modified.insert(&fl.path);
                }
            }
        }

        return Ok(Diff {
            created: created.iter().map(|s| s.to_string()).collect(),
            deleted: deleted.iter().map(|s| s.to_string()).collect(),
            modified: modified.iter().map(|s| s.to_string()).collect(),
        });
    }

    Ok(Diff {
        created: files.keys().cloned().collect(),
        deleted: HashSet::new(),
        modified: HashSet::new(),
    })
}

pub fn load_qdrant_edge() -> Result<EdgeShard, Box<dyn std::error::Error>> {
    let root_path = root_or_cwd()?;
    let p = root_path.join(SHARD_DIRECTORY);
    if p.exists() {
        let edge = EdgeShard::load(&p, Some(edge_config().clone()))?;
        return Ok(edge);
    }
    fs::create_dir_all(&p)?;
    let edge_shard = EdgeShard::new(&p, edge_config().clone())?;
    Ok(edge_shard)
}

/// Create a point struct for upserting.
fn make_point(
    id: uuid::Uuid,
    vector: Vec<f32>,
    code_vector: Vec<f32>,
    sparse_vector: SparseVector,
    payload: Value,
) -> PointStruct {
    PointStruct::new(
        PointId::Uuid(id),
        Vectors::new_named([
            (SPARSE_VECTORS_NAME, Vector::from(sparse_vector)),
            (VECTORS_NAME, Vector::from(vector)),
            (CODE_VECTORS_NAME, Vector::from(code_vector)),
        ]),
        payload,
    )
}

fn upload_embeddings(eps: Vec<EmbeddingWithPayload>) -> Result<(), Box<dyn std::error::Error>> {
    if eps.is_empty() {
        return Ok(());
    }
    let edge = load_qdrant_edge()?;
    let mut points: Vec<PointStructPersisted> = vec![];

    for embd in eps.into_iter() {
        let payload_json = serde_json::to_value(&embd.payload)?;
        let point = make_point(
            uuid::Uuid::new_v4(),
            embd.embedding,
            embd.code_embedding,
            embd.sparse_embedding,
            payload_json,
        );
        points.push(point.into());
    }

    let operation = UpdateOperation::PointOperation(PointOperations::UpsertPoints(
        PointInsertOperations::PointsList(points),
    ));
    edge.update(operation)?;

    Ok(())
}

async fn ingest_one(abs_path: &PathBuf) -> Result<Vec<EmbeddingWithPayload>, String> {
    let ext = abs_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .map(|s| format!(".{}", s.to_lowercase()))
        .unwrap_or_default();
    let content = if SUPPORTED_LIT_EXTENSIONS.contains(&ext.as_str()) {
        let abs_str = abs_path.to_string_lossy().to_string();
        let _guard = PARSER_MUTEX.get_or_init(|| Mutex::new(())).lock().await;
        match parser().parse(&abs_str).await {
            Err(_) => {
                return Err(abs_path.to_string_lossy().to_string());
            }
            Ok(p) => p.text,
        }
    } else {
        match tokio::fs::read_to_string(abs_path).await {
            Err(_) => {
                return Err(abs_path.to_string_lossy().to_string());
            }
            Ok(p) => p,
        }
    };

    let chunks =
        chunk(ext.as_str(), content).map_err(|_| abs_path.to_string_lossy().to_string())?;
    if chunks.is_empty() {
        return Ok(vec![]);
    }
    let chunks = embed(chunks);
    let mut embeddings_with_payloads: Vec<EmbeddingWithPayload> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        embeddings_with_payloads.push(EmbeddingWithPayload {
            embedding: chunk.embedding.unwrap(),
            sparse_embedding: chunk.sparse_embedding.unwrap(),
            code_embedding: chunk.code_embedding.unwrap(),
            payload: EmbeddingPayload {
                document_path: abs_path.to_string_lossy().replace('\\', "/"),
                content: chunk.content,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
            },
        });
    }
    Ok(embeddings_with_payloads)
}

async fn ingest_files(to_ingest: HashSet<String>) -> Result<(), Box<dyn std::error::Error>> {
    if to_ingest.is_empty() {
        return Ok(());
    }
    let root_path = root_or_cwd()?;
    let mut failed = HashSet::new();
    let mut join_set = JoinSet::new();
    let semaphore = Arc::new(Semaphore::new(INGESTION_CONCURRENCY));

    // Channel for streaming embeddings to the upload worker as soon as each file finishes.
    let (tx, mut rx) = mpsc::channel::<Vec<EmbeddingWithPayload>>(INGESTION_CONCURRENCY);
    let embeddings_failed = Arc::new(AtomicUsize::new(0));
    let embeddings_failed_clone = embeddings_failed.clone();

    // Spawn a single task that eagerly uploads embeddings as they arrive.
    let upload_handle = tokio::task::spawn_blocking(move || {
        while let Some(batch) = rx.blocking_recv() {
            if let Err(e) = upload_embeddings(batch) {
                embeddings_failed_clone.fetch_add(1, Ordering::Relaxed);
                eprintln!("✗ Failed to upload embeddings: {e}");
            }
        }
    });

    for fl in to_ingest.iter().progress() {
        let abs_path = root_path.join(fl);
        let permit = semaphore.clone().acquire_owned().await?;
        let tx = tx.clone();
        join_set.spawn(async move {
            let _permit = permit;
            match ingest_one(&abs_path).await {
                Ok(v) => {
                    if !v.is_empty() && tx.send(v).await.is_err() {
                        return Err("Upload channel closed early".into());
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
            Ok(())
        });
    }

    // Drop the original sender so the channel closes once all tasks are done.
    drop(tx);

    let mut panicked = 0;
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Err(e)) => {
                failed.insert(e);
            }
            Err(e) => {
                eprintln!("✗ Error while executing the ingestion function: {e}");
                panicked += 1;
            }
            Ok(Ok(())) => {}
        }
    }

    // Wait for the upload worker to finish draining the channel.
    if let Err(e) = upload_handle.await {
        return Err(format!("Upload worker panicked: {e}").into());
    }

    if embeddings_failed.load(Ordering::Relaxed) != 0 {
        return Err(format!(
            "Failed to upload {:?} embeddings batches",
            embeddings_failed
        )
        .into());
    }

    if panicked != 0 {
        return Err(format!("Panicked while ingesting {:?} files", panicked).into());
    }

    if !failed.is_empty() {
        return Err(format!(
            "Error: failed to ingest {} file(s): {:?}",
            failed.len(),
            failed
        )
        .into());
    }

    Ok(())
}

fn delete_files(to_delete: HashSet<String>) -> Result<(), Box<dyn std::error::Error>> {
    if to_delete.is_empty() {
        return Ok(());
    }
    let edge = load_qdrant_edge()?;
    let condition = Condition::Field(FieldCondition::new_match(
        "document_path"
            .parse::<JsonPath>()
            .map_err(|_| "invalid json path")?,
        Match::from(AnyVariants::Strings(to_delete.iter().cloned().collect())),
    ));
    let filter = Filter::new_must(condition);
    let operation = UpdateOperation::PointOperation(PointOperations::DeletePointsByFilter(filter));
    edge.update(operation)?;

    Ok(())
}

pub fn infer_provider_from_env() -> Result<SupportedProvider, AgentError> {
    for (api_key, base_url, provider_name) in SUPPORT_ENV_VARIABLES {
        if !base_url.is_empty() {
            match std::env::var(base_url) {
                Ok(_) => {
                    if !api_key.is_empty() {
                        match std::env::var(api_key) {
                            Ok(_) => return Ok(SupportedProvider::from_str(provider_name)?),
                            Err(_) => continue,
                        }
                    } else {
                        return Ok(SupportedProvider::from_str(provider_name)?);
                    }
                }
                Err(_) => continue,
            }
        } else {
            match std::env::var(api_key) {
                Ok(_) => return Ok(SupportedProvider::from_str(provider_name)?),
                Err(_) => continue,
            }
        }
    }
    Err(AgentError::ClientInitFailed(
        "Provider could not be resolved from the environment".to_string(),
    ))
}

pub async fn initialize_environment(
    verbose: bool,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let root_path = root_or_cwd()?;
    if !root_path.join(".microagents").exists() {
        fs::create_dir_all(root_path.join(".microagents"))?;
    }

    let _ = edge_config();
    if verbose {
        println!("Loading tokenizer for token estimation...");
    }
    let _ = tokenizer().as_ref()?;

    let files = collect_files()?;
    if verbose {
        println!("Collected all the files in the current directory...");
    }
    let files_content = serde_json::to_string(&files)?;
    let diff = diff_files(files)?;

    if verbose {
        println!(
            "Computed diff for files: re-ingesting {:?} file(s), deleting {:?}",
            diff.to_reingest().len(),
            diff.deleted.len()
        );
    }

    if diff.is_no_diff() {
        if verbose {
            println!("No changes to apply!");
        }
        return Ok((0, 0, 0));
    }

    if verbose {
        println!("Applying changes to detected diff files...");
    }

    if verbose && !diff.deleted.is_empty() {
        println!("Removing deleted files from vector index...")
    }
    delete_files(diff.to_delete())?;
    if verbose && !diff.to_reingest().is_empty() {
        println!("Ingesting changed and added files...")
    }
    ingest_files(diff.to_reingest()).await?;
    persist_file_changes(files_content)?;

    Ok((diff.created.len(), diff.modified.len(), diff.deleted.len()))
}
