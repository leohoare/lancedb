// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Refreshing materialized views.
//!
//! A refresh pins one source version and brings the view to exactly the
//! definition's result at that version. It is incremental when it can
//! reconcile what changed: rows the source added are computed and appended,
//! and rows it removed or changed are evicted by provenance id, the changed
//! ones recomputed into the same pass. A compaction rearranges rows without
//! changing them, so its outputs need no recompute -- this is why a source
//! must keep stable row ids, since skipping that recompute is only sound
//! while [`SOURCE_ROW_ID_COLUMN`] stays valid across the rewrite. A vacuumed
//! watermark version, or an append the classifier cannot separate from a
//! compaction that swallowed it, rebuilds from scratch. Rebuilding an
//! indexed view swaps all fragments in one commit that retains index
//! definitions, so readers never see the view unindexed or empty.
//!
//! The watermark ([`SOURCE_VERSION_META_KEY`]) is stamped in a follow-up
//! commit after the data lands. A crash or race between the two leaves the
//! view visibly unstamped -- its recorded generation no longer matches --
//! and the next refresh rebuilds rather than trusting any of it.
//!
//! Refreshes of one view are serialized within a process by a per-view lock.
//! Across processes the commit serializes them: an incremental refresh
//! carries the provenance ids it materialized as an inserted-rows filter, so
//! two that planned the same rows conflict and only one lands. The loser
//! reports a retryable conflict and replans against the generation that won.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::prelude::{col, lit};
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::mem_wal::DatasetMemWalExt;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::write::delete::DeleteBuilder;
use lance::dataset::write::merge_insert::inserted_rows::{
    KeyExistenceFilter, KeyExistenceFilterBuilder, KeyValue,
};
use lance::dataset::{CommitBuilder, InsertBuilder, WriteDestination, WriteMode, WriteParams};
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID};
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
    /// Rows written to the view: everything on a rebuild, and on an
    /// incremental refresh both the rows added and the rows recomputed in
    /// place of ones the source changed.
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
    full: bool,
    pinned: Option<u64>,
) -> Result<RefreshMaterializedViewResult> {
    let view_native = view.as_native().ok_or_else(|| Error::NotSupported {
        message: "materialized views are supported only on local tables".into(),
    })?;
    view_native.dataset.ensure_mutable()?;
    let lock = refresh_lock(view_native.dataset.get().await?.uri());
    let _guard = lock.lock().await;
    // Force-load the latest view state under the lock: each handle caches
    // lazily, and a second handle would otherwise plan from a snapshot taken
    // before another handle's commit -- appending the same rows again or
    // reporting NoOp over a mutated view.
    view_native.dataset.reload().await?;
    let view_ds = view_native.dataset.get().await?.as_ref().clone();

    // The definition a handle cached at open may since have been replaced;
    // what refresh executes and what it stamps must be one generation.
    let definition = match super::materialized_view_kind(&view_ds.schema().metadata)? {
        Some(super::MaterializedViewKind::Select(definition)) => definition,
        Some(super::MaterializedViewKind::Unrecognized { kind }) => {
            return Err(Error::NotSupported {
                message: format!(
                    "materialized view '{}' is defined by '{kind}', which this \
                     version of lancedb cannot refresh",
                    view.name()
                ),
            });
        }
        None => {
            return Err(Error::NotAMaterializedView {
                name: view.name().to_string(),
            });
        }
    };
    let definition = &definition;
    ensure_no_mem_wal(&view_ds, "materialized view", view.name()).await?;

    let source_ds = open_source(view, definition).await?;
    let source_ds = match pinned {
        Some(version) => source_ds.checkout_version(version).await?,
        None => source_ds,
    };
    ensure_no_mem_wal(&source_ds, "source table", &definition.source_table).await?;
    let source_version = source_ds.version().version;
    let source_ts = source_ds.manifest.timestamp_nanos;

    // Re-plan the persisted definition against the current source schema and
    // require its planned output to be exactly the view's physical schema: a
    // definition the stored table cannot represent must not be certified.
    let source_schema = Arc::new(ArrowSchema::from(source_ds.schema()));
    let projections: Vec<(String, String)> = definition
        .projections
        .iter()
        .map(|p| (p.output.clone(), p.expression.clone()))
        .collect();
    validate_inputs(&source_ds, definition)?;
    let (replanned, mut planned_fields, _renames) = super::plan(
        source_schema,
        &definition.source_table,
        &projections,
        definition.filter.as_deref(),
        definition.limit,
    )?;
    planned_fields.push(arrow_schema::Field::new(
        SOURCE_ROW_ID_COLUMN,
        arrow_schema::DataType::UInt64,
        false,
    ));
    let physical = ArrowSchema::from(view_ds.schema());
    let planned_shape: Vec<_> = planned_fields
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone(), f.is_nullable()))
        .collect();
    let physical_shape: Vec<_> = physical
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone(), f.is_nullable()))
        .collect();
    if planned_shape != physical_shape {
        return Err(Error::Schema {
            message: format!(
                "the stored definition of view '{}' does not produce this \
                 view's schema; recreate the view",
                view.name()
            ),
        });
    }
    let definition = &replanned;

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
        Some(increment) => {
            incremental(
                view_native,
                &view_ds,
                &source_ds,
                source_version,
                source_ts,
                increment,
                definition,
                watermark,
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
) -> Option<Increment> {
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
        // An update rewrites a whole fragment. If it touched one the watermark
        // never saw, that fragment holds rows appended since -- new rows the
        // update did not change, which the recompute does not cover and which
        // this fragment's exclusion from the append set would drop.
        if delta
            .updated_in_place
            .iter()
            .any(|id| !old_ids.contains(id))
        {
            return None;
        }
        // A cap stops rows being materialized once the view is full, and the
        // watermark then advances past them. Eviction can free room those
        // rows should fill, but they are no longer in any delta, so a capped
        // view cannot reconcile a removal incrementally. Rebuilding it is
        // cheap for the same reason it is capped: the scan stops at the cap.
        if definition.limit.is_some() && (delta.deleted_rows || delta.updated_rows) {
            return None;
        }
        // Every other fragment new at head is an append: appends and rewrites
        // are the only operations in the delta that add fragments, and the
        // rewrite outputs are already-materialized rows rearranged.
        return Some(Increment {
            appended: live
                .into_iter()
                .filter(|f| !old_ids.contains(&f.id) && !delta.produced.contains(&f.id))
                .collect(),
            evict_deleted: delta.deleted_rows,
            replace_updated: delta.updated_rows,
        });
    }

    is_pure_append(&old, source_ds, &relevant_field_ids(source_ds, definition)).then(|| Increment {
        appended: live
            .into_iter()
            .filter(|f| !old_ids.contains(&f.id))
            .collect(),
        evict_deleted: false,
        replace_updated: false,
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
/// What an incremental refresh must do to bring the view up to date.
struct Increment {
    /// Source fragments whose rows are not in the view yet.
    appended: Vec<Fragment>,
    /// Source rows left the range, so the view holds rows to evict.
    evict_deleted: bool,
    /// Source rows changed in the range, so the view holds rows to replace.
    replace_updated: bool,
}

struct TxnDelta {
    /// Fragments consumed by `Rewrite` operations.
    rewritten: HashSet<u64>,
    /// Fragments produced by `Rewrite` operations.
    produced: HashSet<u64>,
    /// The delta removed source rows, so the view holds rows to evict.
    deleted_rows: bool,
    /// The delta changed source rows in place, so the view holds rows to
    /// recompute.
    updated_rows: bool,
    /// Fragments an update modified in place, as opposed to produced.
    updated_in_place: HashSet<u64>,
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
        deleted_rows: false,
        updated_rows: false,
        updated_in_place: HashSet::new(),
    };
    for version in (from + 1)..=to {
        let Ok(Some(txn)) = cur.read_transaction_by_version(version).await else {
            return None;
        };
        match txn.operation {
            Operation::Append { .. } | Operation::ReserveFragments { .. } => {}
            // A delete removes source rows without changing the ones that
            // remain, so the view's other rows stay valid: the refresh
            // evicts exactly the ids that left.
            Operation::Delete { .. } => delta.deleted_rows = true,
            // An update changes rows in place, rewriting the fragments that
            // hold them. Those outputs carry no new rows -- the same rows
            // with new values -- so they are excluded from the append set
            // the way a rewrite's outputs are, and the changed rows are
            // replaced individually below.
            Operation::Update {
                removed_fragment_ids,
                new_fragments,
                updated_fragments,
                ..
            } => {
                delta.updated_rows = true;
                // merge_insert reaches here too, and its by-source arm deletes
                // rows rather than changing them.
                delta.deleted_rows = true;
                delta.rewritten.extend(removed_fragment_ids.iter().copied());
                // Only ids of fragments that already existed are real here: a
                // transaction file carries placeholder ids for the fragments
                // it creates, and into an empty source those collide with the
                // ids the commit then assigns. Rewritten rows are excluded by
                // creation version below, not by fragment identity.
                delta
                    .produced
                    .extend(updated_fragments.iter().map(|f| f.id));
                delta
                    .updated_in_place
                    .extend(updated_fragments.iter().map(|f| f.id));
                let _ = new_fragments;
            }
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
    let signature = |fragment: &lance::dataset::fragment::FileFragment| {
        fragment_signature(fragment.metadata(), relevant)
    };
    let current: HashSet<(u64, String)> = cur.get_fragments().iter().map(signature).collect();
    old.get_fragments()
        .iter()
        .all(|fragment| current.contains(&signature(fragment)))
}

/// A fragment's identity as far as the view can observe it: the data files
/// and overlays touching the columns the view reads, plus the deletion file.
/// An overlay replaces cell values without changing any file path, so its
/// identity has to be part of the signature or an overlaid source would
/// classify as a pure append.
fn fragment_signature(metadata: &Fragment, relevant: &HashSet<i32>) -> (u64, String) {
    let touches_relevant =
        |fields: &[i32]| relevant.is_empty() || fields.iter().any(|id| relevant.contains(id));
    let mut files: Vec<&str> = metadata
        .files
        .iter()
        .filter(|file| touches_relevant(&file.fields))
        .map(|file| file.path.as_str())
        .collect();
    files.sort_unstable();
    let mut overlays: Vec<String> = metadata
        .overlays
        .iter()
        .filter(|overlay| touches_relevant(&overlay.data_file.fields))
        .map(|overlay| format!("{}@{}", overlay.data_file.path, overlay.committed_version))
        .collect();
    overlays.sort_unstable();
    (
        metadata.id,
        format!(
            "{}|{}|{:?}",
            files.join(","),
            overlays.join(","),
            metadata.deletion_file
        ),
    )
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

/// Reject MemWAL/LSM state on a refresh participant: rows in un-compacted
/// tiers are visible to ordinary reads but not to the fragment-planned
/// refresh scan, so certifying either side would misrepresent the table.
/// An active write spec and retained rows (the catch-up flag outlives
/// unset) both disqualify.
pub(crate) async fn ensure_no_mem_wal(dataset: &Dataset, role: &str, name: &str) -> Result<()> {
    let retained = dataset.manifest.reader_feature_flags
        & lance_table::feature_flags::FLAG_MEM_WAL_INDEX_CATCHUP
        != 0;
    if retained || dataset.mem_wal_index_details().await?.is_some() {
        return Err(Error::NotSupported {
            message: format!(
                "{role} '{name}' has an LSM write spec or retained un-compacted \
                 rows: rows in un-compacted tiers are invisible to refresh"
            ),
        });
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
    increment: Increment,
    definition: &MaterializedViewDefinition,
    watermark: Option<u64>,
) -> Result<RefreshMaterializedViewResult> {
    let new_fragments = increment.appended;
    let watermark_version = watermark.unwrap_or(0);
    // Rows the source dropped since the watermark. The view holds one row
    // per source row, keyed by provenance, so evicting them is exact: no
    // other view row is affected, which is why a delete no longer forces a
    // rebuild.
    let mut updated_ids: Vec<u64> = Vec::new();
    // Provenance ids whose view rows this refresh removes: rows the source
    // dropped, plus rows it changed, whose current values are recomputed and
    // added back in the same commit.
    let mut gone: Vec<u64> = Vec::new();
    if (increment.evict_deleted || increment.replace_updated)
        && let Some(watermark) = watermark
    {
        let delta = source_ds
            .delta()
            .with_begin_version(watermark)
            .with_end_version(source_version)
            .build()?;
        if increment.evict_deleted {
            let mut stream = delta.get_deleted_row_ids().await?;
            while let Some(batch) = stream.try_next().await? {
                gone.extend(row_ids_of(&batch)?);
            }
        }
        if increment.replace_updated {
            let mut stream = delta.get_updated_rows().await?;
            while let Some(batch) = stream.try_next().await? {
                updated_ids.extend(row_ids_of(&batch)?);
            }
            gone.extend(updated_ids.iter().copied());
        }
    }

    // Staged, not committed: the removals ride in the same commit as the rows
    // that replace them, so a reader never sees the view without either.
    let eviction = if gone.is_empty() {
        None
    } else {
        Some(stage_eviction(view_ds, &gone).await?)
    };

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
    let nothing_to_add =
        (new_fragments.is_empty() && updated_ids.is_empty()) || remaining == Some(0);
    if nothing_to_add && eviction.is_none() {
        result.version =
            stamp_watermark(view_native, view_ds.clone(), source_version, source_ts).await?;
        return Ok(result);
    }
    // Rows left but none arrive: the removals still have to be published.
    if nothing_to_add {
        let published = publish(view_ds, eviction, Vec::new(), None).await?;
        result.version = stamp_watermark(view_native, published, source_version, source_ts).await?;
        return Ok(result);
    }

    // Appends carry the view's schema as it stands; the watermark moves in a
    // follow-up commit (see the module docs for the crash window).
    let schema = Arc::new(ArrowSchema::from(view_ds.schema()));
    let rows_written = Arc::new(AtomicU64::new(0));
    // compute_stream counts what it produces; the truncation below can drop
    // some of that, so the written count comes from the tee instead.
    let computed = Arc::new(AtomicU64::new(0));
    let mut stream = compute_stream(
        source_ds,
        definition,
        RowScope {
            fragments: Some(new_fragments),
            // An update rewrites whole fragments, so a fragment new at head
            // can hold rows the view already has. Their creation version
            // does not change, so it -- not fragment identity -- says which
            // rows are new.
            created_after: increment.replace_updated.then_some(watermark_version),
            limit: remaining,
            ..Default::default()
        },
        schema.clone(),
        computed.clone(),
    )
    .await?;

    // The updated rows' current values, computed the same way and appended
    // in the same commit as the new fragments' rows.
    if !updated_ids.is_empty() {
        let recomputed = compute_stream(
            source_ds,
            definition,
            RowScope {
                row_ids: Some(&updated_ids),
                ..Default::default()
            },
            schema.clone(),
            computed.clone(),
        )
        .await?;
        stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            recomputed.chain(stream),
        ));
    }

    // Nothing survived the filter: the watermark still has to advance or the
    // same fragments would be rescanned forever, but any removals still do.
    let Some(first) = stream.try_next().await? else {
        let published = if eviction.is_some() {
            publish(view_ds, eviction, Vec::new(), None).await?
        } else {
            view_ds.clone()
        };
        result.version = stamp_watermark(view_native, published, source_version, source_ts).await?;
        return Ok(result);
    };
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::iter([Ok(first)]).chain(stream),
    ));

    // Two refreshes that materialize the same source row must not both
    // commit. Carrying the provenance ids as an inserted-rows filter makes
    // lance reject the second on key overlap, while leaving refreshes over
    // disjoint source rows free to proceed.
    let keys = Arc::new(StdMutex::new(KeyExistenceFilterBuilder::new(vec![
        source_row_id_field_id(view_ds)?,
    ])));
    let stream = collect_source_row_ids(stream, keys.clone(), rows_written.clone());

    let ds = Arc::new(view_ds.clone());
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
            message: "expected an append when staging the view's new rows".into(),
        });
    };
    let filter = keys
        .lock()
        .map_err(|_| Error::Runtime {
            message: "the provenance key filter was poisoned mid-refresh".into(),
        })?
        .build();
    let appended = publish(view_ds, eviction, new_fragments, Some(filter)).await?;
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
    let rows_written = Arc::new(AtomicU64::new(0));
    let schema = Arc::new(ArrowSchema::from(view_ds.schema()));
    let stream = compute_stream(
        source_ds,
        definition,
        RowScope {
            limit: definition.limit,
            ..Default::default()
        },
        schema,
        rows_written.clone(),
    )
    .await?;
    // Every rebuild is one fragment swap, indexed or not: an Update commit
    // carries no schema metadata, so it cannot erase a definition update
    // that raced in the way an overwrite (which adopts its stream's schema)
    // durably would -- and it must land on the planned generation or abort.
    let replaced = replace_retaining_indices(view_ds.clone(), stream).await?;
    let version = stamp_watermark(view_native, replaced, source_version, source_ts).await?;
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
    let committed = CommitBuilder::new(WriteDestination::Dataset(ds))
        .execute(transaction)
        .await?;
    if committed.version().version != read_version + 1 {
        return Err(Error::Runtime {
            message: format!(
                "a concurrent commit raced this refresh (view version {}); the \
                 refresh is unrecorded and the next one will rebuild",
                committed.version().version
            ),
        });
    }
    Ok(committed)
}

/// Record that the view now reflects `source_version`, including the view
/// version this very commit produces. The version is predicted and then
/// verified; on a mismatch another commit raced in between, and the stamp
/// ABORTS rather than certify that commit as the refresh's own generation.
/// The view is left visibly unstamped, so the next refresh rebuilds.
async fn stamp_watermark(
    view_native: &NativeTable,
    mut dataset: Dataset,
    source_version: u64,
    source_ts: u128,
) -> Result<u64> {
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
    let actual = dataset.version().version;
    if actual != predicted {
        return Err(Error::Runtime {
            message: format!(
                "a concurrent commit raced this refresh (view version {actual}, \
                 expected {predicted}); the refresh is unrecorded and the next \
                 one will rebuild"
            ),
        });
    }
    view_native.dataset.update(dataset);
    Ok(predicted)
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
/// Which source rows a compute pass reads.
#[derive(Default)]
struct RowScope<'a> {
    /// Read only these fragments.
    fragments: Option<Vec<Fragment>>,
    /// Read only these rows.
    row_ids: Option<&'a [u64]>,
    /// Read only rows created after this version.
    created_after: Option<u64>,
    /// Stop after this many rows.
    limit: Option<u64>,
}

async fn compute_stream(
    source: &Dataset,
    definition: &MaterializedViewDefinition,
    scope: RowScope<'_>,
    schema: SchemaRef,
    rows_written: Arc<AtomicU64>,
) -> Result<SendableRecordBatchStream> {
    let RowScope {
        fragments,
        row_ids,
        created_after,
        limit,
    } = scope;
    let mut scanner = source.scan();
    if let Some(fragments) = fragments {
        scanner.with_fragments(fragments);
    }
    scanner.with_row_id();
    // Narrowing to specific rows keeps the definition's own filter, so a row
    // that no longer satisfies it simply does not come back -- which is how
    // an update that pushes a row out of the view removes it.
    let ids_filter = row_ids.map(|ids| {
        let list = ids
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!("{ROW_ID} IN ({list})")
    });
    let created_filter =
        created_after.map(|version| format!("{ROW_CREATED_AT_VERSION} > {version}"));
    let clauses: Vec<String> = definition
        .filter
        .clone()
        .map(|f| format!("({f})"))
        .into_iter()
        .chain(ids_filter)
        .chain(created_filter)
        .collect();
    if !clauses.is_empty() {
        scanner.filter(&clauses.join(" AND "))?;
    }
    let transforms: Vec<(&str, &str)> = definition
        .projections
        .iter()
        .map(|p| (p.output.as_str(), p.expression.as_str()))
        .collect();
    scanner.project_with_transform(&transforms)?;
    // A scan reads a limit of zero as no limit at all, so a view capped at
    // nothing is answered without one.
    if limit == Some(0) {
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::empty(),
        )));
    }
    if let Some(limit) = limit {
        let limit = i64::try_from(limit).map_err(|_| Error::InvalidInput {
            message: format!("view limit {limit} exceeds the maximum of {}", i64::MAX),
        })?;
        scanner.limit(Some(limit), None)?;
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

/// Commit the view's removals and additions as one change, on the exact
/// generation the refresh planned from. Lance rejects an overlapping
/// provenance key, but an unrelated write to the view is not a key conflict,
/// so the generation is checked here too.
async fn publish(
    view_ds: &Dataset,
    eviction: Option<(Vec<Fragment>, Vec<u64>)>,
    new_fragments: Vec<Fragment>,
    keys: Option<KeyExistenceFilter>,
) -> Result<Dataset> {
    let planned = view_ds.version().version;
    #[cfg(test)]
    tests::hold_before_publish(view_ds.uri()).await;
    let (updated_fragments, removed_fragment_ids) = eviction.unwrap_or_default();
    let committed = CommitBuilder::new(WriteDestination::Dataset(Arc::new(view_ds.clone())))
        .execute(Transaction::new(
            planned,
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified: Vec::new(),
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: Vec::new(),
                update_mode: None,
                inserted_rows_filter: keys,
                updated_fragment_offsets: None,
            },
            None,
        ))
        .await?;
    if committed.version().version != planned + 1 {
        return Err(Error::Runtime {
            message: format!(
                "a concurrent commit raced this refresh (view version {}); \
                 the refresh is unrecorded and the next one will rebuild",
                committed.version().version
            ),
        });
    }
    Ok(committed)
}

fn row_ids_of(batch: &RecordBatch) -> Result<Vec<u64>> {
    let column = batch.column_by_name(ROW_ID).ok_or_else(|| Error::Runtime {
        message: format!("'{ROW_ID}' is missing from a delta batch"),
    })?;
    let ids = column
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| Error::Runtime {
            message: "row ids are not UInt64".into(),
        })?;
    Ok(ids.values().to_vec())
}

/// The fragment changes that remove the view's rows for `ids`, staged rather
/// than committed so they can ride in the refresh's single data commit.
async fn stage_eviction(view_ds: &Dataset, ids: &[u64]) -> Result<(Vec<Fragment>, Vec<u64>)> {
    // An expression rather than SQL text: the id list is a value here, not a
    // predicate string that grows with the delta and has to be parsed.
    let predicate = col(SOURCE_ROW_ID_COLUMN).in_list(
        ids.iter()
            .map(|id| lit(ScalarValue::UInt64(Some(*id))))
            .collect(),
        false,
    );
    let staged = DeleteBuilder::from_expr(Arc::new(view_ds.clone()), predicate)
        .execute_uncommitted()
        .await?;
    let Operation::Delete {
        updated_fragments,
        deleted_fragment_ids,
        ..
    } = staged.transaction.operation
    else {
        return Err(Error::Runtime {
            message: "expected a delete when staging the view's evictions".into(),
        });
    };
    Ok((updated_fragments, deleted_fragment_ids))
}

fn source_row_id_field_id(view_ds: &Dataset) -> Result<i32> {
    view_ds
        .schema()
        .field(SOURCE_ROW_ID_COLUMN)
        .map(|f| f.id)
        .ok_or_else(|| Error::Runtime {
            message: format!("the view has no '{SOURCE_ROW_ID_COLUMN}' column"),
        })
}

/// Tee the provenance ids of everything written into `keys`.
fn collect_source_row_ids(
    stream: SendableRecordBatchStream,
    keys: Arc<StdMutex<KeyExistenceFilterBuilder>>,
    written: Arc<AtomicU64>,
) -> SendableRecordBatchStream {
    let schema = stream.schema();
    let mapped = stream.map(move |batch| {
        let batch = batch?;
        let column = batch
            .column_by_name(SOURCE_ROW_ID_COLUMN)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "'{SOURCE_ROW_ID_COLUMN}' is missing from the rows being written"
                ))
            })?
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!("'{SOURCE_ROW_ID_COLUMN}' is not a uint64"))
            })?;
        let mut keys = keys
            .lock()
            .map_err(|_| DataFusionError::Internal("provenance key filter poisoned".into()))?;
        for id in column.values() {
            keys.insert(KeyValue::UInt64(*id))
                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
        }
        written.fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
        Ok(batch)
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, mapped))
}

#[cfg(test)]
mod tests {

    /// Park a refresh between planning and publication so a test can move
    /// the view underneath it. Inert unless [`DRIFT_TARGET`] names this view.
    pub(super) async fn hold_before_publish(uri: &str) {
        {
            let mut target = DRIFT_TARGET.lock().unwrap();
            if target.as_deref() != Some(uri) {
                return;
            }
            // Take it: memory:// uris are relative and repeat across tests, so
            // leaving it armed would park an unrelated refresh forever.
            *target = None;
        }
        DRIFT_PLANNED.notify_one();
        DRIFT_RELEASED.notified().await;
    }

    /// The rendezvous below is one global pair, so the cases that use it run
    /// one at a time rather than trading each other's signals.
    pub(super) static DRIFT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    pub(super) static DRIFT_TARGET: StdMutex<Option<String>> = StdMutex::new(None);
    pub(super) static DRIFT_PLANNED: tokio::sync::Notify = tokio::sync::Notify::const_new();
    pub(super) static DRIFT_RELEASED: tokio::sync::Notify = tokio::sync::Notify::const_new();
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
    async fn test_update_replaces_the_rows_it_changed() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        // An update changes rows in place, so the view replaces exactly
        // those rows and leaves the rest of what it holds alone.
        source
            .update()
            .column("x", "20")
            .only_if("x = 2")
            .execute()
            .await
            .unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(result.rows_written, 1, "only the changed row is recomputed");
        assert_eq!(read(view.table(), "twice").await, vec![2, 6, 40]);
    }

    #[tokio::test]
    async fn test_delete_evicts_the_view_rows_it_removed() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        // A delete removes source rows without changing the ones that
        // remain, so the view evicts exactly those rows and keeps the rest
        // rather than recomputing every row it already held.
        source.delete("x = 2").await.unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(read(view.table(), "twice").await, vec![2, 6]);

        // A delete and an append in one span: both are applied.
        source.delete("x = 1").await.unwrap();
        source
            .add(record_batch!(("x", Int32, vec![4])).unwrap())
            .execute()
            .await
            .unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(read(view.table(), "twice").await, vec![6, 8]);
    }

    /// merge_insert commits an `Update`, and its by-source arm removes rows
    /// rather than changing them, so the classifier must treat that
    /// transaction form as a source of deletions.
    #[tokio::test]
    async fn test_merge_insert_by_source_delete_evicts_the_view_rows() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        let batch = record_batch!(("x", Int32, vec![1, 3])).unwrap();
        let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let mut merge = source.merge_insert(&["x"]);
        merge.when_not_matched_by_source_delete(None);
        merge.execute(Box::new(reader)).await.unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Incremental);
        assert_eq!(read(view.table(), "twice").await, vec![2, 6]);
    }

    /// A refresh may only certify the generation it planned from. A write
    /// that lands between planning and publication is drift the refresh did
    /// not account for, so it aborts rather than stamp it as materialized.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_refresh_aborts_rather_than_certify_view_drift() {
        let _serial = DRIFT_LOCK.lock().await;
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = conn
            .create_materialized_view("drifting_view", "src")
            .select([("x", "x"), ("twice", "x * 2")])
            .execute()
            .await
            .unwrap();
        view.refresh().execute().await.unwrap();
        let certified = source_version_of(view.table()).await;

        // The source loses a row, so the next refresh plans an eviction.
        source.delete("x = 2").await.unwrap();

        let uri = view
            .table()
            .as_native()
            .unwrap()
            .dataset
            .get()
            .await
            .unwrap()
            .uri()
            .to_string();
        *DRIFT_TARGET.lock().unwrap() = Some(uri);
        let refreshing = tokio::spawn(async move { view.refresh().execute().await });

        // Move the view once the refresh has planned against it.
        tokio::time::timeout(std::time::Duration::from_secs(30), DRIFT_PLANNED.notified())
            .await
            .expect("the refresh never reached the publication boundary");
        let drifted = conn.open_table("drifting_view").execute().await.unwrap();
        drifted.delete("twice = 6").await.unwrap();
        DRIFT_RELEASED.notify_one();

        // Publishing removals and additions as one change makes the drift a
        // conflict lance itself rejects; a pure append, which touches no
        // existing fragment, still relies on the generation check.
        let err = refreshing.await.unwrap().unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("raced this refresh") || message.contains("preempted by concurrent"),
            "got {err:?}"
        );
        // The watermark still names the generation that was actually proven.
        assert_eq!(source_version_of(&drifted).await, certified);
    }

    async fn source_version_of(table: &Table) -> Option<String> {
        table
            .schema()
            .await
            .unwrap()
            .metadata()
            .get(SOURCE_VERSION_META_KEY)
            .cloned()
    }

    /// A transaction file carries placeholder ids for the fragments it
    /// creates. Into an empty source those collide with the ids the commit
    /// assigns, so treating them as already-materialized drops the very
    /// first rows while the watermark still advances past them.
    #[tokio::test]
    async fn test_merge_into_an_empty_source_is_materialized() {
        let (conn, source) = db_with_source(vec![]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();
        assert_eq!(read(view.table(), "twice").await, Vec::<i32>::new());

        let batch = record_batch!(("x", Int32, vec![1, 2])).unwrap();
        let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let mut merge = source.merge_insert(&["x"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(Box::new(reader)).await.unwrap();

        view.refresh().execute().await.unwrap();
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
        // A second refresh must not double them either.
        view.refresh().execute().await.unwrap();
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
    }

    /// A refresh publishes what it removes and what it adds as one change,
    /// so an update never exposes the view without the rows it is replacing.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_an_update_is_never_visible_as_a_gap() {
        let _serial = DRIFT_LOCK.lock().await;
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = conn
            .create_materialized_view("atomic_view", "src")
            .select([("x", "x"), ("twice", "x * 2")])
            .execute()
            .await
            .unwrap();
        view.refresh().execute().await.unwrap();

        source
            .update()
            .column("x", "x + 10")
            .execute()
            .await
            .unwrap();

        let uri = view
            .table()
            .as_native()
            .unwrap()
            .dataset
            .get()
            .await
            .unwrap()
            .uri()
            .to_string();
        *DRIFT_TARGET.lock().unwrap() = Some(uri);
        let refreshing = tokio::spawn(async move { view.refresh().execute().await });

        // Read the view while the refresh is staged but not yet published.
        tokio::time::timeout(std::time::Duration::from_secs(30), DRIFT_PLANNED.notified())
            .await
            .expect("the refresh never reached the publication boundary");
        let midway = conn.open_table("atomic_view").execute().await.unwrap();
        assert_eq!(
            read(&midway, "twice").await,
            vec![2, 4, 6],
            "the pre-refresh rows must still be there in full"
        );
        DRIFT_RELEASED.notify_one();

        refreshing.await.unwrap().unwrap();
        let after = conn.open_table("atomic_view").execute().await.unwrap();
        assert_eq!(read(&after, "twice").await, vec![22, 24, 26]);
    }

    /// A cap of zero is a view that holds nothing, not a view without a cap.
    #[tokio::test]
    async fn test_zero_limit_holds_no_rows() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = conn
            .create_materialized_view("empty", "src")
            .select([("x", "x")])
            .limit(0)
            .execute()
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.rows_written, 0);
        assert_eq!(view.table().count_rows(None).await.unwrap(), 0);

        // Still nothing after the source grows, and the watermark advances.
        append(&source, vec![4]).await;
        view.refresh().execute().await.unwrap();
        assert_eq!(view.table().count_rows(None).await.unwrap(), 0);
        assert_eq!(
            view.refresh().execute().await.unwrap().mode,
            RefreshMode::NoOp
        );
    }

    /// An update rewrites a whole fragment. When it lands on one appended
    /// since the watermark, that fragment also holds rows the update never
    /// touched -- rows the recompute does not cover and the append set no
    /// longer reaches.
    #[tokio::test]
    async fn test_update_touching_a_new_fragment_keeps_its_untouched_rows() {
        let (conn, source) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        // One fragment, appended after the watermark, holding both a row the
        // update will change and a row it will not.
        append(&source, vec![3, 40]).await;
        source
            .update()
            .column("x", "99")
            .only_if("x = 3")
            .execute()
            .await
            .unwrap();

        view.refresh().execute().await.unwrap();
        assert_eq!(
            read(view.table(), "twice").await,
            vec![2, 4, 80, 198],
            "a row appended into the updated fragment went missing"
        );
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

    /// A cap and incremental reconciliation do not compose: rows skipped at
    /// the cap fall behind the watermark, so room freed later cannot be
    /// refilled from any delta. A capped view rebuilds instead, which its
    /// own cap keeps cheap.
    #[tokio::test]
    async fn test_limited_view_rebuilds_rather_than_reconcile() {
        let (conn, source) = db_with_source(vec![1, 2]).await;
        let view = conn
            .create_materialized_view("capped", "src")
            .select([("x", "x")])
            .limit(2)
            .execute()
            .await
            .unwrap();
        view.refresh().execute().await.unwrap();
        assert_eq!(read(view.table(), "x").await, vec![1, 2]);

        append(&source, vec![3, 4]).await;
        source
            .update()
            .column("x", "11")
            .only_if("x = 1")
            .execute()
            .await
            .unwrap();
        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        let held = read(view.table(), "x").await;
        assert_eq!(held.len(), 2, "the cap holds: {held:?}");
        let selectable = read(&source, "x").await;
        assert!(
            held.iter().all(|x| selectable.contains(x)),
            "{held:?} is not a subset of {selectable:?}"
        );
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

    /// A second handle's lazy cache must not defeat the lock: after another
    /// handle's refresh commits, the stale handle plans from the reloaded
    /// state and no-ops instead of appending the same fragments again.
    #[tokio::test]
    async fn test_a_second_handle_does_not_double_append() {
        let (conn, source) = db_with_source(vec![1, 2, 3]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();
        let stale = conn.open_materialized_view("doubled").await.unwrap();

        append(&source, vec![4]).await;
        view.refresh().execute().await.unwrap();

        let result = stale.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::NoOp);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4, 6, 8]);
    }

    /// A commit racing between a refresh's data commit and its stamp must
    /// not be certified as the refresh's generation: the stamp aborts, and
    /// the next refresh rebuilds from the drifted state.
    #[tokio::test]
    async fn test_stamp_aborts_on_a_racing_commit() {
        let (conn, _) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        let view_native = view.table().as_native().unwrap();
        let stale = view_native.dataset.get().await.unwrap().as_ref().clone();
        view.table().delete("x = 1").await.unwrap();

        let err = stamp_watermark(view_native, stale, 99, 99).await;
        assert!(err.is_err());

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(read(view.table(), "twice").await, vec![2, 4]);
    }

    /// What refresh executes and what it stamps must be one generation: a
    /// replaced definition wins over whatever a stale handle cached.
    #[tokio::test]
    async fn test_refresh_uses_the_latest_persisted_definition() {
        let (conn, _) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        let replacement = crate::materialized_view::MaterializedViewDefinition {
            source_table: "src".into(),
            projections: vec![
                crate::materialized_view::ViewProjection {
                    output: "x".into(),
                    expression: "x".into(),
                },
                crate::materialized_view::ViewProjection {
                    output: "twice".into(),
                    expression: "x * 3".into(),
                },
            ],
            filter: None,
            limit: None,
            inputs: vec!["x".into()],
        };
        let mut metadata = HashMap::new();
        metadata.insert(
            crate::materialized_view::DEFINITION_META_KEY.to_string(),
            crate::materialized_view::definition_to_metadata(&replacement).unwrap(),
        );
        view.table()
            .as_native()
            .unwrap()
            .replace_schema_metadata(metadata)
            .await
            .unwrap();

        let result = view.refresh().execute().await.unwrap();
        assert_eq!(result.mode, RefreshMode::Rebuild);
        assert_eq!(read(view.table(), "twice").await, vec![3, 6]);
    }

    /// A persisted definition that does not produce this view's schema must
    /// not refresh at all, let alone be certified.
    #[tokio::test]
    async fn test_definition_view_schema_mismatch_is_refused() {
        let (conn, _) = db_with_source(vec![1, 2]).await;
        let view = doubled_view(&conn).await;
        view.refresh().execute().await.unwrap();

        let narrower = crate::materialized_view::MaterializedViewDefinition {
            source_table: "src".into(),
            projections: vec![crate::materialized_view::ViewProjection {
                output: "x".into(),
                expression: "x".into(),
            }],
            filter: None,
            limit: None,
            inputs: vec!["x".into()],
        };
        let mut metadata = HashMap::new();
        metadata.insert(
            crate::materialized_view::DEFINITION_META_KEY.to_string(),
            crate::materialized_view::definition_to_metadata(&narrower).unwrap(),
        );
        view.table()
            .as_native()
            .unwrap()
            .replace_schema_metadata(metadata)
            .await
            .unwrap();

        let err = view.refresh().execute().await.unwrap_err();
        assert!(matches!(err, Error::Schema { message } if message.contains("does not produce")),);
    }

    /// An overlay replaces cell values without changing any file path; the
    /// signature must see it, scoped to the columns the view reads like
    /// data files are.
    #[test]
    fn test_fragment_signature_sees_overlays() {
        use lance_file::version::ConcreteFileVersion;
        use lance_table::format::DataFile;
        use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};

        let base = Fragment::new(7);
        let mut file = DataFile::new_unstarted("f0.lance", ConcreteFileVersion::V2_1);
        file.fields = vec![0, 1].into();
        let mut with_file = base.clone();
        with_file.files.push(file.clone());

        let overlay = |field: i32| {
            let mut data_file = DataFile::new_unstarted("o0.lance", ConcreteFileVersion::V2_1);
            data_file.fields = vec![field].into();
            DataOverlayFile {
                data_file,
                coverage: OverlayCoverage::PerField(Vec::new()),
                committed_version: 9,
            }
        };
        let relevant: HashSet<i32> = [0].into_iter().collect();

        let mut overlaid_relevant = with_file.clone();
        overlaid_relevant.overlays.push(overlay(0));
        assert_ne!(
            fragment_signature(&with_file, &relevant),
            fragment_signature(&overlaid_relevant, &relevant),
        );

        let mut overlaid_unrelated = with_file.clone();
        overlaid_unrelated.overlays.push(overlay(5));
        assert_eq!(
            fragment_signature(&with_file, &relevant),
            fragment_signature(&overlaid_unrelated, &relevant),
        );
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

    /// MemWAL tiers are visible to reads but not to the refresh scan, so LSM
    /// state disqualifies every participant: the source at create, either
    /// side at refresh, and the view can never accept a spec. Retained
    /// un-compacted rows (the catch-up flag outlives unset) count as state.
    #[tokio::test]
    async fn lsm_state_disqualifies_source_and_view() {
        use crate::table::LsmWriteSpec;
        use arrow_array::RecordBatchIterator;

        let tmp_dir = tempfile::tempdir().unwrap();
        let conn = connect(tmp_dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();
        // Hand-rolled: the LSM primary key must be non-nullable, which
        // record_batch! cannot express.
        let schema = Arc::new(ArrowSchema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false),
            arrow_schema::Field::new("x", arrow_schema::DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![1, 2])) as _,
                Arc::new(Int32Array::from(vec![1, 2])) as _,
            ],
        )
        .unwrap();
        let table = conn
            .create_table("src", batch.clone())
            .write_options(crate::materialized_view::tests::stable_row_ids())
            .execute()
            .await
            .unwrap();
        table.set_unenforced_primary_key(["id"]).await.unwrap();
        table
            .set_lsm_write_spec(LsmWriteSpec::unsharded())
            .await
            .unwrap();

        // An active-LSM source is refused at create.
        let err = conn
            .create_materialized_view("v", "src")
            .execute()
            .await
            .unwrap_err();
        assert!(err.to_string().contains("un-compacted"), "{err}");

        // Unset with nothing written clears the state: the view creates,
        // and a spec can never be installed over it.
        table.unset_lsm_write_spec().await.unwrap();
        let view = conn
            .create_materialized_view("v", "src")
            .execute()
            .await
            .unwrap();
        let err = view
            .table()
            .set_lsm_write_spec(LsmWriteSpec::unsharded())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("materialized view"), "{err}");

        // A source that acquires a spec after create fails refresh.
        table
            .set_lsm_write_spec(LsmWriteSpec::unsharded())
            .await
            .unwrap();
        let err = view.refresh().execute().await.unwrap_err();
        assert!(err.to_string().contains("source table 'src'"), "{err}");

        // Retained rows outlive unset: write through the WAL, unset, and
        // refresh still refuses on the catch-up flag.
        table.require_mem_wal_index_catchup().await.unwrap();
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all()
            .use_lsm(true);
        merge
            .execute(Box::new(RecordBatchIterator::new(
                vec![Ok(batch.clone())],
                batch.schema(),
            )))
            .await
            .unwrap();
        table.unset_lsm_write_spec().await.unwrap();
        let err = view.refresh().execute().await.unwrap_err();
        assert!(err.to_string().contains("source table 'src'"), "{err}");
    }
}
