// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Refreshing materialized views.
//!
//! A refresh pins one source version and brings the view to exactly the
//! definition's result at that version. It is incremental -- computing only
//! the source fragments that carry new rows, and appending -- when the source
//! changed by nothing but appends and compactions since the last refresh: a
//! compaction rearranges rows without changing them, so its outputs need no
//! recompute (this is why a source must keep stable row ids -- skipping the
//! recompute is only sound while [`SOURCE_ROW_ID_COLUMN`] stays valid across
//! the rewrite). A delete, an update, a vacuumed watermark version, or an
//! append the classifier cannot separate from a compaction that swallowed it
//! rebuilds the view from scratch. Rebuilding an indexed view swaps all
//! fragments in one commit that retains index definitions, so readers never
//! see the view unindexed or empty.
//!
//! The watermark ([`SOURCE_VERSION_META_KEY`]) is stamped after the data
//! commit, not atomically with it, except on the unindexed rebuild path where
//! it rides the overwrite. A crash between the two can leave the watermark
//! behind the data; the next incremental refresh would then re-append rows it
//! already holds.
//!
//! Refreshes of one view are serialized within a process by a per-view lock.
//! Across processes nothing serializes them, and two incremental refreshes
//! planned at the same watermark would each append the same rows; run one
//! process's refreshes against a view at a time.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::RecordBatch;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::error::DataFusionError;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, InsertBuilder, WriteDestination, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::ROW_ID;
use lance_table::format::Fragment;
use serde::{Deserialize, Serialize};

use super::{
    MaterializedViewDefinition, REFRESHED_AT_MS_META_KEY, SOURCE_ROW_ID_COLUMN,
    SOURCE_VERSION_META_KEY,
};
use crate::database::OpenTableRequest;
use crate::table::{NativeTable, NativeTableExt, Table};
use crate::{Error, Result};

/// How a refresh brought the view up to date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefreshMode {
    /// The view was recomputed from scratch.
    Rebuild,
    /// Rows from source fragments added since the last refresh were appended.
    Incremental,
    /// The view was already at the requested source version.
    NoOp,
}

/// The result of refreshing a materialized view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefreshMaterializedViewResult {
    /// How the view was brought up to date.
    pub mode: RefreshMode,
    /// Rows written to the view: everything on a rebuild, the appended rows
    /// on an incremental refresh.
    pub rows_written: u64,
    /// The source table version the view now reflects.
    pub source_version: u64,
    /// The view table version after the refresh.
    pub version: u64,
}

/// Schema metadata key holding the view table version a successful refresh
/// left behind. Any other commit on the view is drift, and refresh rebuilds.
pub const VIEW_VERSION_META_KEY: &str = "mv.view_version";

/// Schema metadata key holding the commit timestamp of the watermark's source
/// manifest. A dropped and recreated source reuses version numbers but never
/// their timestamps, so a mismatch means the watermark describes a different
/// incarnation and refresh rebuilds.
pub const SOURCE_VERSION_TS_META_KEY: &str = "mv.source_version_ts";

/// One refresh per view at a time within this process.
fn refresh_lock(uri: &str) -> Arc<tokio::sync::Mutex<()>> {
    static LOCKS: OnceLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
        OnceLock::new();
    LOCKS
        .get_or_init(Default::default)
        .lock()
        .expect("refresh lock registry poisoned")
        .entry(uri.to_string())
        .or_default()
        .clone()
}

/// Internal implementation of the refresh logic.
pub(crate) async fn execute_refresh(
    view: &Table,
    definition: &MaterializedViewDefinition,
    full: bool,
    pinned: Option<u64>,
) -> Result<RefreshMaterializedViewResult> {
    let view_native = view.as_native().ok_or_else(|| Error::NotSupported {
        message: "materialized views are supported only on local tables".into(),
    })?;
    view_native.dataset.ensure_mutable()?;
    let lock = refresh_lock(view_native.dataset.get().await?.uri());
    let _guard = lock.lock().await;
    // Snapshot the view under the lock, so a concurrent refresh's commits
    // are either fully visible here or fully ordered after this one.
    let view_ds = view_native.dataset.get().await?.as_ref().clone();

    let source_ds = open_source(view, definition).await?;
    let source_ds = match pinned {
        Some(version) => source_ds.checkout_version(version).await?,
        None => source_ds,
    };
    let source_version = source_ds.version().version;
    let source_ts = source_ds.manifest.timestamp_nanos;

    validate_inputs(&source_ds, definition)?;

    let metadata = &view_ds.schema().metadata;
    let watermark: Option<u64> = metadata
        .get(SOURCE_VERSION_META_KEY)
        .and_then(|raw| raw.parse().ok());
    let recorded_ts: Option<u128> = metadata
        .get(SOURCE_VERSION_TS_META_KEY)
        .and_then(|raw| raw.parse().ok());
    // The watermark speaks only for the view state its refresh left behind;
    // any other commit on the view since then is drift.
    let view_intact = metadata
        .get(VIEW_VERSION_META_KEY)
        .and_then(|raw| raw.parse::<u64>().ok())
        == Some(view_ds.version().version);

    if !full && watermark == Some(source_version) && view_intact && recorded_ts == Some(source_ts) {
        return Ok(RefreshMaterializedViewResult {
            mode: RefreshMode::NoOp,
            rows_written: 0,
            source_version,
            version: view_ds.version().version,
        });
    }

    let watermark = watermark.filter(|_| view_intact);
    match plan_increment(
        &source_ds,
        source_version,
        watermark,
        recorded_ts,
        full,
        definition,
    )
    .await
    {
        Some(new_fragments) => {
            incremental(
                view_native,
                &view_ds,
                &source_ds,
                source_version,
                source_ts,
                new_fragments,
                definition,
            )
            .await
        }
        None => {
            rebuild(
                view_native,
                &view_ds,
                &source_ds,
                source_version,
                source_ts,
                definition,
            )
            .await
        }
    }
}

/// The source fragments whose rows are new since the watermark, or `None`
/// where the view has to rebuild: no watermark yet, an explicit full refresh,
/// a source that moved backwards, a vacuumed watermark version, or a change
/// the classifier cannot prove left every already-materialized row intact.
///
/// Two tiers. The transaction walk is exact where it applies: an
/// appends-and-compactions delta computes precisely the appended fragments,
/// and compaction outputs -- the same rows rearranged -- cost nothing. The
/// fragment-signature check is the fallback for deltas the walk cannot read
/// (an unknown operation, a missing transaction file, a version gap past
/// [`MAX_TRANSACTION_WALK`]); it cannot tell a compaction from a rewrite of
/// row content, so under it any fragment churn rebuilds.
async fn plan_increment(
    source_ds: &Dataset,
    source_version: u64,
    watermark: Option<u64>,
    recorded_ts: Option<u128>,
    full: bool,
    definition: &MaterializedViewDefinition,
) -> Option<Vec<Fragment>> {
    if full {
        return None;
    }
    let watermark = watermark?;
    if watermark > source_version {
        return None;
    }
    let old = source_ds.checkout_version(watermark).await.ok()?;
    // A recreated source reuses version numbers, never their timestamps: a
    // mismatch means the watermark describes a different incarnation.
    if recorded_ts != Some(old.manifest.timestamp_nanos) {
        return None;
    }
    let old_ids: HashSet<u64> = old.get_fragments().iter().map(|f| f.id() as u64).collect();
    let live: Vec<Fragment> = source_ds
        .get_fragments()
        .iter()
        .map(|f| f.metadata().clone())
        .collect();

    if let Some(delta) = appends_and_rewrites(source_ds, watermark, source_version).await {
        // A rewrite that consumed a fragment neither present at the watermark
        // nor produced by an earlier rewrite swallowed a mid-delta append;
        // its rows cannot be told apart from already-materialized ones.
        let folded = delta
            .rewritten
            .iter()
            .any(|id| !old_ids.contains(id) && !delta.produced.contains(id));
        if folded {
            return None;
        }
        // Every other fragment new at head is an append: appends and rewrites
        // are the only operations in the delta that add fragments, and the
        // rewrite outputs are already-materialized rows rearranged.
        return Some(
            live.into_iter()
                .filter(|f| !old_ids.contains(&f.id) && !delta.produced.contains(&f.id))
                .collect(),
        );
    }

    is_pure_append(&old, source_ds, &relevant_field_ids(source_ds, definition)).then(|| {
        live.into_iter()
            .filter(|f| !old_ids.contains(&f.id))
            .collect()
    })
}

/// Version gap beyond which per-version transaction reads stop being cheaper
/// than one fragment scan.
const MAX_TRANSACTION_WALK: u64 = 512;

/// Fragment ids moved by the `Rewrite` operations of an
/// appends-and-compactions delta.
///
/// Only rewrite ids are collected: a transaction file records an `Append`'s
/// fragments with placeholder ids (they are assigned at commit), while a
/// rewrite's ids are reserved before it commits and are real. Appends are
/// therefore derived as what is new at head and not a rewrite output.
struct TxnDelta {
    /// Fragments consumed by `Rewrite` operations.
    rewritten: HashSet<u64>,
    /// Fragments produced by `Rewrite` operations.
    produced: HashSet<u64>,
}

/// Read the delta from the transaction log, `None` where it holds anything
/// but appends and rewrites -- or cannot be read at all. `None` can only send
/// the caller to a slower check, never change the answer.
///
/// `ReserveFragments` is compaction reserving its output ids; it moves no
/// rows and rides along.
async fn appends_and_rewrites(cur: &Dataset, from: u64, to: u64) -> Option<TxnDelta> {
    if to <= from || to - from > MAX_TRANSACTION_WALK {
        return None;
    }
    let mut delta = TxnDelta {
        rewritten: HashSet::new(),
        produced: HashSet::new(),
    };
    for version in (from + 1)..=to {
        let Ok(Some(txn)) = cur.read_transaction_by_version(version).await else {
            return None;
        };
        match txn.operation {
            Operation::Append { .. } | Operation::ReserveFragments { .. } => {}
            Operation::Rewrite { groups, .. } => {
                for group in groups {
                    delta
                        .rewritten
                        .extend(group.old_fragments.iter().map(|f| f.id));
                    delta
                        .produced
                        .extend(group.new_fragments.iter().map(|f| f.id));
                }
            }
            _ => return None,
        }
    }
    Some(delta)
}

/// Fallback pure-append check: every fragment of the old version is still
/// present with an identical signature -- same data files touching the
/// columns the view reads, same deletion file. Compaction replaces fragment
/// ids, a delete adds or changes a deletion file, an update rewrites data
/// files: each breaks the subset relation and forces a rebuild. A new data
/// file for a column the view does not read leaves the signature alone,
/// which is what lets this tier pass deltas the transaction walk cannot
/// (a backfill of an unrelated column, an added all-null column).
fn is_pure_append(old: &Dataset, cur: &Dataset, relevant: &HashSet<i32>) -> bool {
    let signature = |fragment: &lance::dataset::fragment::FileFragment| -> (u64, String) {
        let metadata = fragment.metadata();
        let mut files: Vec<&str> = metadata
            .files
            .iter()
            .filter(|file| {
                relevant.is_empty() || file.fields.iter().any(|id| relevant.contains(id))
            })
            .map(|file| file.path.as_str())
            .collect();
        files.sort_unstable();
        (
            fragment.id() as u64,
            format!("{}|{:?}", files.join(","), metadata.deletion_file),
        )
    };
    let current: HashSet<(u64, String)> = cur.get_fragments().iter().map(signature).collect();
    old.get_fragments()
        .iter()
        .all(|fragment| current.contains(&signature(fragment)))
}

/// Field ids (with struct descendants) of the source columns the view reads.
fn relevant_field_ids(source: &Dataset, definition: &MaterializedViewDefinition) -> HashSet<i32> {
    fn collect(field: &lance_core::datatypes::Field, ids: &mut HashSet<i32>) {
        ids.insert(field.id);
        for child in &field.children {
            collect(child, ids);
        }
    }
    let mut ids = HashSet::new();
    for input in &definition.inputs {
        if let Some(field) = source.schema().field(input) {
            collect(field, &mut ids);
        }
    }
    ids
}

/// Error if a column the view reads no longer exists in the source.
fn validate_inputs(source: &Dataset, definition: &MaterializedViewDefinition) -> Result<()> {
    for input in &definition.inputs {
        if source.schema().field(input).is_none() {
            return Err(Error::Schema {
                message: format!(
                    "source column '{input}' read by the view no longer exists \
                     (dropped or renamed in '{}')",
                    definition.source_table
                ),
            });
        }
    }
    Ok(())
}

async fn open_source(view: &Table, definition: &MaterializedViewDefinition) -> Result<Dataset> {
    let database = view.database_opt().ok_or_else(|| Error::InvalidInput {
        message: "the view was not opened through a database connection".into(),
    })?;
    let source = database
        .open_table(OpenTableRequest {
            name: definition.source_table.clone(),
            namespace_path: Vec::new(),
            index_cache_size: None,
            lance_read_params: None,
            location: None,
            namespace_client: None,
            managed_versioning: None,
        })
        .await?;
    let native = source.as_native().ok_or_else(|| Error::NotSupported {
        message: "materialized views are supported only on local tables".into(),
    })?;
    let dataset = native.dataset.get().await?.as_ref().clone();
    if !dataset.manifest.uses_stable_row_ids() {
        return Err(Error::InvalidInput {
            message: format!(
                "source table '{}' does not have stable row ids; it is not the \
                 table this view was declared over",
                definition.source_table
            ),
        });
    }
    Ok(dataset)
}

#[allow(clippy::too_many_arguments)]
async fn incremental(
    view_native: &NativeTable,
    view_ds: &Dataset,
    source_ds: &Dataset,
    source_version: u64,
    source_ts: u128,
    new_fragments: Vec<Fragment>,
    definition: &MaterializedViewDefinition,
) -> Result<RefreshMaterializedViewResult> {
    // The cap counts rows already materialized, in first-materialized order.
    let remaining = match definition.limit {
        Some(limit) => {
            let held = view_ds.count_rows(None).await? as u64;
            Some(limit.saturating_sub(held))
        }
        None => None,
    };

    let mut result = RefreshMaterializedViewResult {
        mode: RefreshMode::Incremental,
        rows_written: 0,
        source_version,
        version: view_ds.version().version,
    };
    if new_fragments.is_empty() || remaining == Some(0) {
        result.version =
            stamp_watermark(view_native, view_ds.clone(), source_version, source_ts).await?;
        return Ok(result);
    }

    // Appends carry the view's schema as it stands; the watermark moves in a
    // follow-up commit (see the module docs for the crash window).
    let schema = Arc::new(ArrowSchema::from(view_ds.schema()));
    let rows_written = Arc::new(AtomicU64::new(0));
    let mut stream = compute_stream(
        source_ds,
        definition,
        Some(new_fragments),
        remaining,
        schema.clone(),
        rows_written.clone(),
    )
    .await?;

    // Nothing survived the filter: the watermark still has to advance or the
    // same fragments would be rescanned forever.
    let Some(first) = stream.try_next().await? else {
        result.version =
            stamp_watermark(view_native, view_ds.clone(), source_version, source_ts).await?;
        return Ok(result);
    };
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::iter([Ok(first)]).chain(stream),
    ));

    let appended = InsertBuilder::new(WriteDestination::Dataset(Arc::new(view_ds.clone())))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_stream(stream)
        .await?;
    result.rows_written = rows_written.load(Ordering::Relaxed);
    result.version = stamp_watermark(view_native, appended, source_version, source_ts).await?;
    Ok(result)
}

async fn rebuild(
    view_native: &NativeTable,
    view_ds: &Dataset,
    source_ds: &Dataset,
    source_version: u64,
    source_ts: u128,
    definition: &MaterializedViewDefinition,
) -> Result<RefreshMaterializedViewResult> {
    let indexed = !view_ds.load_indices().await?.is_empty();
    let rows_written = Arc::new(AtomicU64::new(0));

    let new_dataset = if indexed {
        let schema = Arc::new(ArrowSchema::from(view_ds.schema()));
        let stream = compute_stream(
            source_ds,
            definition,
            None,
            definition.limit,
            schema,
            rows_written.clone(),
        )
        .await?;
        let replaced = replace_retaining_indices(view_ds.clone(), stream).await?;
        // The swap keeps the previous schema metadata, watermark included, so
        // it is restamped in a follow-up commit.
        let version = stamp_watermark(view_native, replaced, source_version, source_ts).await?;
        return Ok(RefreshMaterializedViewResult {
            mode: RefreshMode::Rebuild,
            rows_written: rows_written.load(Ordering::Relaxed),
            source_version,
            version,
        });
    } else {
        // An overwrite adopts the stream's schema, so the watermark rides the
        // same commit as the data.
        let mut metadata = view_ds.schema().metadata.clone();
        metadata.insert(
            SOURCE_VERSION_META_KEY.to_string(),
            source_version.to_string(),
        );
        metadata.insert(
            SOURCE_VERSION_TS_META_KEY.to_string(),
            source_ts.to_string(),
        );
        metadata.insert(REFRESHED_AT_MS_META_KEY.to_string(), now_ms().to_string());
        // The overwrite is the refresh's final commit when nothing races it;
        // the stamped view version is verified below and re-stamped if not.
        let predicted = view_ds.version().version + 1;
        metadata.insert(VIEW_VERSION_META_KEY.to_string(), predicted.to_string());
        let schema = Arc::new(ArrowSchema::from(view_ds.schema()).with_metadata(metadata));
        let stream = compute_stream(
            source_ds,
            definition,
            None,
            definition.limit,
            schema,
            rows_written.clone(),
        )
        .await?;
        InsertBuilder::new(WriteDestination::Dataset(Arc::new(view_ds.clone())))
            .with_params(&WriteParams {
                mode: WriteMode::Overwrite,
                enable_stable_row_ids: true,
                ..Default::default()
            })
            .execute_stream(stream)
            .await?
    };

    let version = new_dataset.version().version;
    let recorded: Option<u64> = new_dataset
        .schema()
        .metadata
        .get(VIEW_VERSION_META_KEY)
        .and_then(|raw| raw.parse().ok());
    let version = if recorded == Some(version) {
        view_native.dataset.update(new_dataset);
        version
    } else {
        stamp_watermark(view_native, new_dataset, source_version, source_ts).await?
    };
    Ok(RefreshMaterializedViewResult {
        mode: RefreshMode::Rebuild,
        rows_written: rows_written.load(Ordering::Relaxed),
        source_version,
        version,
    })
}

/// Replace all of the view's data in one commit that retains its index
/// definitions.
///
/// An overwrite resets the manifest's indices, so an indexed view would
/// briefly report no index; a delete-then-append pair would briefly show
/// zero rows. Writing the new fragments uncommitted and committing one
/// `Update` that removes every old fragment does neither: `Update` prunes
/// index bitmaps only for modified fields and none are modified here, so the
/// index definitions survive, covering zero rows until the new ones are
/// reindexed (searches brute-force them meanwhile).
async fn replace_retaining_indices(
    view_ds: Dataset,
    stream: SendableRecordBatchStream,
) -> Result<Dataset> {
    let ds = Arc::new(view_ds);
    let read_version = ds.version().version;
    let removed_fragment_ids: Vec<u64> = ds.get_fragments().iter().map(|f| f.id() as u64).collect();

    let write_txn = InsertBuilder::new(WriteDestination::Dataset(ds.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted_stream(stream)
        .await?;
    let Operation::Append {
        fragments: new_fragments,
    } = write_txn.operation
    else {
        return Err(Error::Runtime {
            message: "expected an append when staging the view's replacement rows".into(),
        });
    };

    let transaction = Transaction::new(
        read_version,
        Operation::Update {
            removed_fragment_ids,
            updated_fragments: Vec::new(),
            new_fragments,
            fields_modified: Vec::new(),
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        },
        None,
    );
    Ok(CommitBuilder::new(WriteDestination::Dataset(ds))
        .execute(transaction)
        .await?)
}

/// Record that the view now reflects `source_version`, including the view
/// version this very commit produces (predicted, then verified, so a racing
/// commit cannot leave the record pointing at someone else's version), and
/// hand the updated dataset to the table handle. Returns the view version.
async fn stamp_watermark(
    view_native: &NativeTable,
    mut dataset: Dataset,
    source_version: u64,
    source_ts: u128,
) -> Result<u64> {
    for _ in 0..3 {
        let predicted = dataset.version().version + 1;
        dataset
            .update_schema_metadata([
                (
                    SOURCE_VERSION_META_KEY.to_string(),
                    Some(source_version.to_string()),
                ),
                (
                    SOURCE_VERSION_TS_META_KEY.to_string(),
                    Some(source_ts.to_string()),
                ),
                (
                    REFRESHED_AT_MS_META_KEY.to_string(),
                    Some(now_ms().to_string()),
                ),
                (
                    VIEW_VERSION_META_KEY.to_string(),
                    Some(predicted.to_string()),
                ),
            ])
            .await?;
        if dataset.version().version == predicted {
            view_native.dataset.update(dataset);
            return Ok(predicted);
        }
    }
    Err(Error::Runtime {
        message: "concurrent writes kept moving the view while recording its refresh".into(),
    })
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Evaluate the definition over `source`, restricted to `fragments` when
/// given, as a stream of batches in the view's schema.
///
/// The filter is pushed into the scan; each projection is carried as a
/// `(name, expression)` projection pair, never spliced into SQL text. Each
/// view field is filled by the projection of the same name, and
/// [`SOURCE_ROW_ID_COLUMN`] by the scan's row id.
async fn compute_stream(
    source: &Dataset,
    definition: &MaterializedViewDefinition,
    fragments: Option<Vec<Fragment>>,
    limit: Option<u64>,
    schema: SchemaRef,
    rows_written: Arc<AtomicU64>,
) -> Result<SendableRecordBatchStream> {
    let mut scanner = source.scan();
    if let Some(fragments) = fragments {
        scanner.with_fragments(fragments);
    }
    scanner.with_row_id();
    if let Some(filter) = &definition.filter {
        scanner.filter(filter)?;
    }
    let transforms: Vec<(&str, &str)> = definition
        .projections
        .iter()
        .map(|p| (p.output.as_str(), p.expression.as_str()))
        .collect();
    scanner.project_with_transform(&transforms)?;
    if let Some(limit) = limit {
        scanner.limit(Some(limit as i64), None)?;
    }

    let out_schema = schema.clone();
    let mapped = scanner.try_into_stream().await?.map(move |batch| {
        let batch = batch.map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut columns = Vec::with_capacity(out_schema.fields().len());
        for field in out_schema.fields() {
            let name = if field.name() == SOURCE_ROW_ID_COLUMN {
                ROW_ID
            } else {
                field.name()
            };
            let column = batch.column_by_name(name).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "view column '{}' is not produced by the view's definition",
                    field.name()
                ))
            })?;
            columns.push(column.clone());
        }
        rows_written.fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
        Ok(RecordBatch::try_new(out_schema.clone(), columns)?)
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, mapped)))
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int32Array, record_batch};
    use futures::TryStreamExt;
    use lance::dataset::NewColumnTransform;

    use super::*;
    use crate::connect;
    use crate::connection::Connection;
    use crate::index::Index;
    use crate::index::scalar::BTreeIndexBuilder;
    use crate::materialized_view::MaterializedView;
    use crate::query::{ExecutableQuery, QueryBase, Select};
    use crate::table::{CompactionOptions, OptimizeAction};

    async fn db_with_source(values: Vec<i32>) -> (Connection, Table) {
        let conn = connect("memory://").execute().await.unwrap();
        let batch = record_batch!(("x", Int32, values)).unwrap();
        let table = conn
            .create_table("src", batch)
            .write_options(crate::materialized_view::tests::stable_row_ids())
            .execute()
            .await
            .unwrap();
        (conn, table)
    }

    async fn doubled_view(conn: &Connection) -> MaterializedView {
        conn.create_materialized_view("doubled", "src")
            .select([("x", "x"), ("twice", "x * 2")])
            .execute()
            .await
            .unwrap()
    }

    async fn read(table: &Table, column: &str) -> Vec<i32> {
        let batches = table
            .query()
            .select(Select::columns(&[column]))
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut values: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch[column]
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .collect();
        values.sort();
        values
    }

    async fn append(table: &Table, values: Vec<i32>) {
        let batch = record_batch!(("x", Int32, values)).unwrap();
        table.add(batch).execute().await.unwrap();
    }

    #[tokio::test]
    async fn test_first_refresh_materializes_the_view() {
        let (conn, _) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(result.rows_written, 3);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4, 6]);

        // The watermark survives on the stored schema, not just the handle.
        let reopened = conn.open_materialized_view("doubled").await.unwrap();
        let again = reopened.refresh().execute().await.unwrap();
        assert_eq!(again.mode, RefreshMode::NoOp);
        assert_eq!(again.rows_written, 0);
    }

    #[tokio::test]
    async fn test_filter_selects_the_source_rows() {
        let (conn, _) = db_with_source(vec![1, 20, 3, 40]).await;
        let view = conn
            .create_materialized_view("big", "src")
            .select([("x", "x")])
            .only_if("x > 10")
            .execute()
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.rows_written, 2);
        assert_eq!(read(view.table(), "x").await, vec![20, 40]);
    }

    #[tokio::test]
    async fn test_append_refreshes_incrementally() {
        let (conn, source) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        append(&source, vec![5]).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 1);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4, 10]);
    }

    #[tokio::test]
    async fn test_incremental_applies_the_filter() {
        let (conn, source) = db_with_source(vec![1, 20]).await;
        let view = conn
            .create_materialized_view("big", "src")
            .select([("x", "x")])
            .only_if("x > 10")
            .execute()
            .await
            .unwrap();
        view.refresh().execute().await.unwrap();

        append(&source, vec![3, 30]).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 1);
        assert_eq!(read(view.table(), "x").await, vec![20, 30]);
    }

    /// The watermark has to advance even when no appended row matches, or the
    /// same fragments would be rescanned by every later refresh.
    #[tokio::test]
    async fn test_incremental_with_nothing_matching_advances_the_watermark() {
        let (conn, source) = db_with_source(vec![20]).await;
        let view = conn
            .create_materialized_view("big", "src")
            .select([("x", "x")])
            .only_if("x > 10")
            .execute()
            .await
            .unwrap();
        view.refresh().execute().await.unwrap();

        append(&source, vec![1, 2]).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 0);

        let again = view.refresh().execute().await.unwrap();
        assert_eq!(again.mode, RefreshMode::NoOp);
    }

    /// Unlike a computed column, a view reflects source mutation: an update
    /// rebuilds rather than going stale.
    #[tokio::test]
    async fn test_update_forces_a_rebuild() {
        let (conn, source) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        source.update().column("x", "100").execute().await.unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(read(view.table(), "twice").await, vec![200, 200]);
    }

    #[tokio::test]
    async fn test_delete_forces_a_rebuild() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        source.delete("x = 2").await.unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(read(view.table(), "twice").await, vec![2, 6]);
    }

    async fn compact(source: &Table) {
        source
            .optimize(OptimizeAction::Compact {
                options: CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .unwrap();
    }

    /// A compaction rearranges rows without changing them, so it costs the
    /// view nothing: the watermark advances and no row is recomputed.
    #[tokio::test]
    async fn test_compaction_alone_refreshes_incrementally() {
        let (conn, source) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();
        append(&source, vec![3]).await;
        view.refresh().execute().await.unwrap();

        compact(&source).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 0);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4, 6]);
        assert_eq!(
            view.refresh().execute().await.unwrap().mode,
            RefreshMode::NoOp
        );
    }

    /// Rows appended after a compaction are separable through the transaction
    /// log: only the appended fragments are computed.
    #[tokio::test]
    async fn test_append_after_compaction_stays_incremental() {
        let (conn, source) = db_with_source(vec![1]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();
        append(&source, vec![2]).await;
        view.refresh().execute().await.unwrap();

        compact(&source).await;
        append(&source, vec![3]).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 1);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4, 6]);
    }

    /// An append swallowed by a later compaction cannot be told apart from
    /// the rows the view already holds, so the refresh rebuilds -- once --
    /// rather than duplicate or drop.
    #[tokio::test]
    async fn test_append_folded_into_compaction_rebuilds() {
        let (conn, source) = db_with_source(vec![1]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        append(&source, vec![2]).await;
        compact(&source).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
    }

    /// The create-time gate holds across a drop-and-recreate of the source
    /// under the same name.
    #[tokio::test]
    async fn test_refresh_refuses_a_recreated_source_without_stable_row_ids() {
        let (conn, _) = db_with_source(vec![1]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        conn.drop_table("src", &[]).await.unwrap();
        let batch = record_batch!(("x", Int32, [9])).unwrap();
        conn.create_table("src", batch).execute().await.unwrap();

        let err = view.refresh().execute().await.unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { message } if message.contains("stable row ids"))
        );
    }

    /// A change to a column the view does not read is not a reason to
    /// rebuild: the exact signature check is scoped to the view's inputs.
    #[tokio::test]
    async fn test_unrelated_column_change_does_not_rebuild() {
        let (conn, source) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        source
            .add_columns()
            .transform(NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(
                vec![arrow_schema::Field::new(
                    "unrelated",
                    arrow_schema::DataType::Int32,
                    true,
                )],
            ))))
            .execute()
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 0);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
    }

    #[tokio::test]
    async fn test_full_forces_a_rebuild() {
        let (conn, source) = db_with_source(vec![1]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        append(&source, vec![2]).await;
        let result = view.refresh().full(true).execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(result.rows_written, 2);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
    }

    #[tokio::test]
    async fn test_limit_caps_the_view() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = conn
            .create_materialized_view("capped", "src")
            .select([("x", "x")])
            .limit(4)
            .execute()
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.rows_written, 3);

        // The cap counts already-held rows, so only one appended row lands.
        append(&source, vec![4, 5, 6]).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 1);
        assert_eq!(view.table().count_rows(None).await.unwrap(), 4);

        // At the cap, later appends only move the watermark.
        append(&source, vec![7]).await;
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.rows_written, 0);
        assert_eq!(
            view.refresh().execute().await.unwrap().mode,
            RefreshMode::NoOp
        );
    }

    /// The whole point of the fragment-swap commit: a rebuild must never
    /// leave the view without its index definitions.
    #[tokio::test]
    async fn test_rebuild_retains_indexes() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        view.table()
            .create_index(&["twice"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await
            .unwrap();
        assert_eq!(view.table().list_indices().await.unwrap().len(), 1);

        source
            .update()
            .column("x", "x + 10")
            .execute()
            .await
            .unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(view.table().list_indices().await.unwrap().len(), 1);
        assert_eq!(read(view.table(), "twice").await, vec![22, 24, 26]);

        // The swapped-in rows are reachable through an indexed query.
        let batches = view
            .table()
            .query()
            .only_if("twice = 24")
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    }

    #[tokio::test]
    async fn test_rebuild_of_an_empty_result_is_an_empty_view() {
        let (conn, _) = db_with_source(vec![1, 2]).await;
        let view = conn
            .create_materialized_view("none", "src")
            .select([("x", "x")])
            .only_if("x > 100")
            .execute()
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(result.rows_written, 0);
        assert_eq!(view.table().count_rows(None).await.unwrap(), 0);
        assert_eq!(
            view.refresh().execute().await.unwrap().mode,
            RefreshMode::NoOp
        );
    }

    /// Provenance: every view row records the source row that produced it.
    #[tokio::test]
    async fn test_source_row_ids_are_recorded() {
        let (conn, _) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        let batches = view
            .table()
            .query()
            .select(Select::columns(&[SOURCE_ROW_ID_COLUMN]))
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
        for batch in &batches {
            assert_eq!(batch[SOURCE_ROW_ID_COLUMN].null_count(), 0);
        }
    }

    #[tokio::test]
    async fn test_dropping_a_source_input_fails_the_refresh() {
        let (conn, source) = db_with_source(vec![1]).await;
        let view = conn
            .create_materialized_view("v", "src")
            .select([("twice", "x * 2")])
            .execute()
            .await
            .unwrap();
        view.refresh().execute().await.unwrap();

        append(&source, vec![2]).await;
        source
            .add_columns()
            .transform(NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(
                vec![arrow_schema::Field::new(
                    "y",
                    arrow_schema::DataType::Int32,
                    true,
                )],
            ))))
            .execute()
            .await
            .unwrap();
        source.drop_columns(&["x"]).await.unwrap();

        let err = view.refresh().execute().await.unwrap_err();
        assert!(matches!(err, Error::Schema { message } if message.contains("'x'")));
    }

    /// A pinned refresh materializes the source as of `version`; catching up
    /// to the appends beyond it stays incremental.
    #[tokio::test]
    async fn test_pinned_refresh_and_catch_up() {
        let (conn, source) = db_with_source(vec![1]).await;
        let view = doubled_view(&conn).await;
        let pinned = source.version().await.unwrap();

        append(&source, vec![2]).await;
        let result = view
            .refresh()
            .source_version(pinned)
            .execute()
            .await
            .unwrap();
        assert_eq!(result.source_version, pinned);
        assert_eq!(read(view.table(), "twice").await, vec![2]);

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
    }

    /// Views chain: a view's stable row ids and provenance column make it a
    /// source like any other, and the default projection takes its declared
    /// columns without copying its provenance.
    #[tokio::test]
    async fn test_a_view_can_source_another_view() {
        let (conn, source) = db_with_source(vec![1, 2, 30]).await;
        let first = doubled_view(&conn).await;
        first.refresh().execute().await.unwrap();

        let second = conn
            .create_materialized_view("second", "doubled")
            .only_if("twice > 10")
            .execute()
            .await
            .unwrap();
        assert!(
            second
                .definition()
                .projections
                .iter()
                .all(|p| p.output != SOURCE_ROW_ID_COLUMN)
        );
        let result = second.refresh().execute().await.unwrap();
        assert_eq!(result.rows_written, 1);
        assert_eq!(read(second.table(), "twice").await, vec![60]);

        append(&source, vec![50]).await;
        first.refresh().execute().await.unwrap();
        let result = second.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(read(second.table(), "twice").await, vec![60, 100]);
    }

    /// The watermark speaks only for the state a refresh left behind: a
    /// direct write to the view is drift, and the next refresh rebuilds
    /// rather than preserving it as current.
    #[tokio::test]
    async fn test_direct_view_mutation_forces_a_rebuild() {
        let (conn, _) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        view.table().delete("x = 1").await.unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
        assert_eq!(
            view.refresh().execute().await.unwrap().mode,
            RefreshMode::NoOp
        );
    }

    /// A dropped and recreated source reuses version numbers but never their
    /// timestamps; the watermark must not vouch for the replacement's rows.
    #[tokio::test]
    async fn test_source_recreation_forces_a_rebuild() {
        let (conn, _) = db_with_source(vec![1]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        conn.drop_table("src", &[]).await.unwrap();
        let batch = record_batch!(("x", Int32, [7])).unwrap();
        conn.create_table("src", batch)
            .write_options(crate::materialized_view::tests::stable_row_ids())
            .execute()
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(read(view.table(), "twice").await, vec![14]);
    }

    /// In-process refreshes of one view serialize: the loser of the race
    /// observes the winner's watermark instead of appending the same rows.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_refreshes_do_not_duplicate() {
        let (conn, source) = db_with_source(vec![1]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        append(&source, vec![2, 3]).await;
        let (a, b) = tokio::join!(view.refresh().execute(), view.refresh().execute());
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!(read(view.table(), "twice").await, vec![2, 4, 6]);
        let modes = [a.mode, b.mode];
        assert!(modes.contains(&RefreshMode::Incremental));
        assert!(modes.contains(&RefreshMode::NoOp));
    }

    /// An output whose name needs quoting flows through as a projection
    /// alias, never spliced into SQL text.
    #[tokio::test]
    async fn test_output_names_needing_quotes() {
        let (conn, _) = db_with_source(vec![1, 2]).await;
        let view = conn
            .create_materialized_view("v", "src")
            .select([("double value", "x * 2")])
            .execute()
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.rows_written, 2);
        assert_eq!(read(view.table(), "double value").await, vec![2, 4]);
    }
}
