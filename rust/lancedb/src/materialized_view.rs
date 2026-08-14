// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Materialized views.
//!
//! A materialized view is a normal table whose contents are defined by a query
//! over one source table -- projected expressions, an optional filter, an
//! optional row limit -- and maintained by refresh rather than by writes.
//! Creating one commits an empty table carrying the definition in schema
//! metadata; [`MaterializedView::refresh`] computes the rows. The source
//! table must have stable row ids ([`CreateMaterializedViewBuilder::execute`]
//! says why).
//!
//! The definition is tagged by kind so a variant added later reads back as a
//! view this version cannot refresh, not as a plain table. Only the projected
//! `select` form exists today.
//!
//! Because the view is a table, everything a table supports -- queries,
//! indexes, search -- works on it unchanged. Writes are not blocked, but a
//! refresh that rebuilds replaces them; the definition is the source of truth.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use lance_core::ROW_ID;
use lance_datafusion::planner::Planner;
use serde::{Deserialize, Serialize};

use crate::connection::Connection;
use crate::table::refresh::quote_identifier;
use crate::table::{Table, WriteOptions};
use crate::{Error, Result};

/// Schema metadata key holding the view definition, as kind-tagged JSON.
pub const DEFINITION_META_KEY: &str = "mv.definition";

/// Schema metadata key holding the source table version the view was last
/// refreshed to. Absent until the first refresh.
pub const SOURCE_VERSION_META_KEY: &str = "mv.source_version";

/// Schema metadata key holding the wall-clock time of the last refresh,
/// in milliseconds since the epoch.
pub const REFRESHED_AT_MS_META_KEY: &str = "mv.refreshed_at_ms";

/// Column recording which source row produced each view row.
///
/// Values are the source's stable `_rowid` at refresh time, valid across
/// source compactions, updates and deletes -- which is why a view's source
/// is required to keep stable row ids.
pub const SOURCE_ROW_ID_COLUMN: &str = "__source_row_id";

/// Value of the definition's `kind` tag for the projected `select` form.
pub const SELECT_KIND: &str = "select";

/// One projected output column of a view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewProjection {
    /// Name of the column in the view.
    pub output: String,
    /// SQL expression over the source table that computes it.
    pub expression: String,
}

/// The query that defines a materialized view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializedViewDefinition {
    /// Name of the source table, in the same database as the view.
    pub source_table: String,
    /// The projected output columns, in view schema order.
    pub projections: Vec<ViewProjection>,
    /// SQL predicate selecting the source rows the view holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// Cap on the number of rows the view holds, in materialization order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    /// Source columns the projections and filter read, derived at creation.
    #[serde(default)]
    pub inputs: Vec<String>,
}

/// A view definition as read back from schema metadata.
///
/// Non-exhaustive: a kind added later is an additive change, and a caller that
/// only handles the kinds it knows keeps compiling.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MaterializedViewKind {
    /// The projected `select` form.
    Select(MaterializedViewDefinition),
    /// A kind this version does not understand, written by a newer one.
    ///
    /// Reported rather than hidden so a caller can tell a view it cannot
    /// refresh apart from a plain table. Nothing produces this.
    Unrecognized {
        /// The kind as it was found in the metadata.
        kind: String,
    },
}

/// Serialize `definition` into the kind-tagged form stored under
/// [`DEFINITION_META_KEY`].
pub(crate) fn definition_to_metadata(definition: &MaterializedViewDefinition) -> Result<String> {
    let mut value = serde_json::to_value(definition).map_err(|e| Error::Runtime {
        message: format!("failed to serialize view definition: {e}"),
    })?;
    value["kind"] = serde_json::Value::String(SELECT_KIND.to_string());
    Ok(value.to_string())
}

/// Read a view declaration off a schema metadata map, if it carries one.
///
/// `Ok(None)` for a plain table. A present declaration that does not parse is
/// an error rather than `None`: the table was a view, and treating it as plain
/// would let it be silently rewritten.
pub fn materialized_view_kind(
    metadata: &HashMap<String, String>,
) -> Result<Option<MaterializedViewKind>> {
    let Some(raw) = metadata.get(DEFINITION_META_KEY) else {
        return Ok(None);
    };
    let unreadable = |e: &dyn std::fmt::Display| Error::Runtime {
        message: format!("unreadable materialized view definition: {e}"),
    };
    let value: serde_json::Value = serde_json::from_str(raw).map_err(|e| unreadable(&e))?;
    let kind = value
        .get("kind")
        .and_then(|k| k.as_str())
        .ok_or_else(|| unreadable(&"missing kind tag"))?;
    if kind != SELECT_KIND {
        return Ok(Some(MaterializedViewKind::Unrecognized {
            kind: kind.to_string(),
        }));
    }
    let definition = serde_json::from_value(value).map_err(|e| unreadable(&e))?;
    Ok(Some(MaterializedViewKind::Select(definition)))
}

/// Resolve a definition against the source schema into the view's projected
/// fields, with `inputs` filled in.
///
/// Everything that can be known statically is checked here rather than at
/// refresh time: that each expression and the filter parse, that every column
/// they read exists, and that output names are free and unreserved. Empty
/// `projections` selects every source column, expanded against the schema as
/// it is now -- columns added to the source later are not picked up.
pub(crate) fn plan(
    source_schema: SchemaRef,
    source_table: &str,
    projections: &[(String, String)],
    filter: Option<&str>,
    limit: Option<u64>,
) -> Result<(MaterializedViewDefinition, Vec<ArrowField>)> {
    let projections: Vec<(String, String)> = if projections.is_empty() {
        source_schema
            .fields()
            .iter()
            // A source that is itself a view carries its own provenance
            // column; the new view records its own, not a copy.
            .filter(|f| f.name() != SOURCE_ROW_ID_COLUMN)
            .map(|f| (f.name().clone(), quote_identifier(f.name())))
            .collect()
    } else {
        projections.to_vec()
    };

    let planner = Planner::new(source_schema.clone());
    let mut fields = Vec::with_capacity(projections.len());
    let mut inputs = Vec::new();
    let mut declared: Vec<&str> = Vec::with_capacity(projections.len());

    for (output, expression) in &projections {
        if declared.contains(&output.as_str()) {
            return Err(Error::ColumnAlreadyExists {
                name: output.clone(),
            });
        }
        if output == SOURCE_ROW_ID_COLUMN || output == ROW_ID {
            return Err(Error::InvalidInput {
                message: format!("view column name '{output}' is reserved"),
            });
        }

        let parsed = planner
            .parse_expr(expression)
            .map_err(|e| Error::InvalidExpression {
                column: output.clone(),
                message: e.to_string(),
            })?;
        // Before optimization: the simplifier folds a stable-but-not-immutable
        // call like now() into a literal, hiding it from the check while the
        // stored definition keeps the call.
        ensure_immutable(&parsed, |message| Error::InvalidExpression {
            column: output.clone(),
            message,
        })?;
        let expr = planner
            .optimize_expr(parsed)
            .map_err(|e| Error::InvalidExpression {
                column: output.clone(),
                message: e.to_string(),
            })?;
        let expr_inputs =
            resolve_inputs(&source_schema, &expr, |message| Error::InvalidExpression {
                column: output.clone(),
                message,
            })?;

        // Physical expressions address columns by position, so the planner
        // that types the expression is built on the projected schema.
        let read_schema = project_schema(&source_schema, &expr_inputs);
        let physical = Planner::new(read_schema.clone())
            .create_physical_expr(&expr)
            .map_err(|e| Error::InvalidExpression {
                column: output.clone(),
                message: e.to_string(),
            })?;
        let data_type =
            physical
                .data_type(read_schema.as_ref())
                .map_err(|e| Error::InvalidExpression {
                    column: output.clone(),
                    message: e.to_string(),
                })?;

        // Always nullable: what a refresh appends must fit the declared field
        // whatever nullability the evaluator reports for a given batch.
        fields.push(ArrowField::new(output, data_type, true));
        inputs.extend(expr_inputs);
        declared.push(output);
    }

    if let Some(filter) = filter {
        let expr = planner
            .parse_filter(filter)
            .map_err(|e| Error::InvalidInput {
                message: format!("invalid view filter: {e}"),
            })?;
        ensure_immutable(&expr, |message| Error::InvalidInput {
            message: format!("invalid view filter: {message}"),
        })?;
        let filter_inputs = resolve_inputs(&source_schema, &expr, |message| Error::InvalidInput {
            message: format!("invalid view filter: {message}"),
        })?;
        // A committed filter has to be usable as a predicate.
        let read_schema = project_schema(&source_schema, &filter_inputs);
        let data_type = Planner::new(read_schema.clone())
            .create_physical_expr(&expr)
            .map_err(|e| Error::InvalidInput {
                message: format!("invalid view filter: {e}"),
            })?
            .data_type(read_schema.as_ref())
            .map_err(|e| Error::InvalidInput {
                message: format!("invalid view filter: {e}"),
            })?;
        if data_type != DataType::Boolean {
            return Err(Error::InvalidInput {
                message: format!("view filter must be a boolean predicate, not {data_type}"),
            });
        }
        inputs.extend(filter_inputs);
    }

    inputs.sort();
    inputs.dedup();

    let definition = MaterializedViewDefinition {
        source_table: source_table.to_string(),
        projections: projections
            .into_iter()
            .map(|(output, expression)| ViewProjection { output, expression })
            .collect(),
        filter: filter.map(String::from),
        limit,
        inputs,
    };
    Ok((definition, fields))
}

/// Reject any function that is not immutable: a view definition has to
/// evaluate identically across refreshes, or incremental maintenance would
/// mix rows from different evaluations of the same definition.
fn ensure_immutable(expr: &datafusion_expr::Expr, error: impl Fn(String) -> Error) -> Result<()> {
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_expr::Volatility;

    // Labeled immutable but not determined by row values alone: version()
    // depends on the build, and the arrow_* introspectors on schema state
    // (name, type, nullability, field metadata) that may change without any
    // data change a refresh could observe.
    const NOT_VALUE_DETERMINED: &[&str] =
        &["version", "arrow_typeof", "arrow_field", "arrow_metadata"];

    let mut offending: Option<String> = None;
    expr.apply(|node| {
        if let datafusion_expr::Expr::ScalarFunction(function) = node {
            let name = function.func.name();
            if function.func.signature().volatility != Volatility::Immutable
                || NOT_VALUE_DETERMINED.contains(&name)
            {
                offending = Some(name.to_string());
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map_err(|e| error(e.to_string()))?;
    match offending {
        Some(name) => Err(error(format!(
            "function '{name}' is not immutable and would evaluate differently \
             across refreshes"
        ))),
        None => Ok(()),
    }
}

/// The root of a possibly-dotted column path: `metadata.age` -> `metadata`.
fn root(path: &str) -> &str {
    path.split('.').next().unwrap_or(path)
}

/// The columns `expr` reads, kept as the planner reports them (a nested
/// reference stays a dotted path) but resolved by root field.
fn resolve_inputs(
    schema: &ArrowSchema,
    expr: &datafusion_expr::Expr,
    error: impl Fn(String) -> Error,
) -> Result<Vec<String>> {
    let mut inputs = Planner::column_names_in_expr(expr);
    inputs.sort();
    inputs.dedup();
    for input in &inputs {
        if schema.field_with_name(root(input)).is_err() {
            return Err(error(format!("unknown column '{input}'")));
        }
    }
    Ok(inputs)
}

/// Project the root fields of `columns`, deduplicated, in schema order.
fn project_schema(schema: &ArrowSchema, columns: &[String]) -> SchemaRef {
    let roots: std::collections::HashSet<&str> = columns.iter().map(|c| root(c)).collect();
    let fields: Vec<ArrowField> = schema
        .fields()
        .iter()
        .filter(|f| roots.contains(f.name().as_str()))
        .map(|f| f.as_ref().clone())
        .collect();
    Arc::new(ArrowSchema::new(fields))
}

/// One row of [`Connection::list_materialized_views`]: a view's name and its
/// definition kind, which may be one this version cannot refresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedViewEntry {
    /// Name of the view's table.
    pub name: String,
    /// The view's definition as stored.
    pub kind: MaterializedViewKind,
}

/// Materialized views are local-only; refuse a remote connection before any
/// request is made.
fn ensure_local(connection: &Connection) -> Result<()> {
    if connection.uri().starts_with("db://") {
        return Err(Error::NotSupported {
            message: "materialized views are supported only on local databases".into(),
        });
    }
    Ok(())
}

/// Builds a materialized view. Created by
/// [`Connection::create_materialized_view`].
pub struct CreateMaterializedViewBuilder {
    connection: Connection,
    name: String,
    source: String,
    projections: Vec<(String, String)>,
    filter: Option<String>,
    limit: Option<u64>,
}

impl CreateMaterializedViewBuilder {
    pub(crate) fn new(connection: Connection, name: String, source: String) -> Self {
        Self {
            connection,
            name,
            source,
            projections: Vec::new(),
            filter: None,
            limit: None,
        }
    }

    /// The view's columns, as `(name, SQL expression)` pairs.
    ///
    /// Not calling this selects every source column, expanded against the
    /// source schema at creation time.
    pub fn select(
        mut self,
        columns: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.projections = columns
            .into_iter()
            .map(|(output, expression)| (output.into(), expression.into()))
            .collect();
        self
    }

    /// Only source rows matching the SQL predicate appear in the view.
    pub fn only_if(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Cap the view at `limit` rows, in materialization order.
    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Create the view: an empty table carrying the definition. No rows are
    /// computed until a refresh.
    ///
    /// The source table must have stable row ids: they are what keeps the
    /// view's provenance valid across source compactions, updates and
    /// deletes, and they cannot be enabled after a table exists.
    pub async fn execute(self) -> Result<MaterializedView> {
        ensure_local(&self.connection)?;
        let source = self.connection.open_table(&self.source).execute().await?;
        let Some(native) = source.as_native() else {
            return Err(Error::NotSupported {
                message: "materialized views are supported only on local databases".into(),
            });
        };
        if !native.dataset.get().await?.manifest.uses_stable_row_ids() {
            return Err(Error::InvalidInput {
                message: format!(
                    "materialized views require stable row ids on the source table; \
                     create '{}' with storage option new_table_enable_stable_row_ids=true",
                    self.source
                ),
            });
        }
        let source_schema = source.schema().await?;
        let (definition, mut fields) = plan(
            source_schema,
            &self.source,
            &self.projections,
            self.filter.as_deref(),
            self.limit,
        )?;
        fields.push(ArrowField::new(
            SOURCE_ROW_ID_COLUMN,
            DataType::UInt64,
            false,
        ));

        let metadata = HashMap::from([(
            DEFINITION_META_KEY.to_string(),
            definition_to_metadata(&definition)?,
        )]);
        let schema = Arc::new(ArrowSchema::new_with_metadata(fields, metadata));
        // Stable row ids so the view can itself source another view.
        let table = self
            .connection
            .create_empty_table(&self.name, schema)
            .write_options(WriteOptions {
                lance_write_params: Some(lance::dataset::WriteParams {
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            })
            .execute()
            .await?;
        let stable = match table.as_native() {
            Some(native) => native.dataset.get().await?.manifest.uses_stable_row_ids(),
            None => false,
        };
        if !stable {
            let _ = self.connection.drop_table(&self.name, &[]).await;
            return Err(Error::InvalidInput {
                message: format!(
                    "view '{}' would be created without stable row ids: the \
                     connection's new_table_enable_stable_row_ids setting \
                     overrides view creation; remove it or use a connection \
                     without it",
                    self.name
                ),
            });
        }
        Ok(MaterializedView { table, definition })
    }
}

/// A handle on a materialized view: the view table plus its parsed definition.
#[derive(Debug, Clone)]
pub struct MaterializedView {
    table: Table,
    definition: MaterializedViewDefinition,
}

impl MaterializedView {
    /// Interpret `table` as a materialized view.
    ///
    /// Fails with [`Error::NotAMaterializedView`] for a plain table and
    /// [`Error::NotSupported`] for a view whose kind this version cannot
    /// refresh.
    pub async fn from_table(table: Table) -> Result<Self> {
        let schema = table.schema().await?;
        match materialized_view_kind(schema.metadata())? {
            Some(MaterializedViewKind::Select(definition)) => Ok(Self { table, definition }),
            Some(MaterializedViewKind::Unrecognized { kind }) => Err(Error::NotSupported {
                message: format!(
                    "materialized view '{}' is defined by '{kind}', which this version of \
                     lancedb cannot refresh",
                    table.name()
                ),
            }),
            None => Err(Error::NotAMaterializedView {
                name: table.name().to_string(),
            }),
        }
    }

    /// The view, as the table it is. Queries, indexes and search all apply.
    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn name(&self) -> &str {
        self.table.name()
    }

    /// The query that defines the view.
    pub fn definition(&self) -> &MaterializedViewDefinition {
        &self.definition
    }
}

impl Connection {
    /// Define a materialized view named `name` over `source`.
    ///
    /// The view is created empty, with the definition recorded in its schema
    /// metadata; refresh computes the rows. Local databases only.
    ///
    /// ```no_run
    /// # use lancedb::Connection;
    /// # async fn create(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    /// let view = conn
    ///     .create_materialized_view("loud_adults", "people")
    ///     .select([("name", "upper(name)"), ("age", "age")])
    ///     .only_if("age >= 18")
    ///     .execute()
    ///     .await?;
    /// println!("{}", view.definition().source_table);
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_materialized_view(
        &self,
        name: impl Into<String>,
        source: impl Into<String>,
    ) -> CreateMaterializedViewBuilder {
        CreateMaterializedViewBuilder::new(self.clone(), name.into(), source.into())
    }

    /// Open the materialized view named `name`.
    pub async fn open_materialized_view(
        &self,
        name: impl Into<String>,
    ) -> Result<MaterializedView> {
        ensure_local(self)?;
        let table = self.open_table(name).execute().await?;
        MaterializedView::from_table(table).await
    }

    /// The materialized views in this database, unrefreshable kinds
    /// included: a view written by a newer version lists with its kind
    /// preserved rather than disappearing.
    ///
    /// Reads every table's schema to find them, so this costs an open per
    /// table. A table that cannot be opened is skipped rather than failing
    /// the listing.
    pub async fn list_materialized_views(&self) -> Result<Vec<MaterializedViewEntry>> {
        ensure_local(self)?;
        let names = self.table_names().execute().await?;
        let mut views = Vec::new();
        for name in names {
            let Ok(table) = self.open_table(&name).execute().await else {
                continue;
            };
            let schema = table.schema().await?;
            if let Some(kind) = materialized_view_kind(schema.metadata())? {
                views.push(MaterializedViewEntry { name, kind });
            }
        }
        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::record_batch;

    use super::*;
    use crate::connect;
    use crate::table::WriteOptions;

    async fn people_db() -> Connection {
        let conn = connect("memory://").execute().await.unwrap();
        let batch = record_batch!(
            ("name", Utf8, ["ada", "grace", "alan"]),
            ("age", Int32, [36, 85, 41])
        )
        .unwrap();
        conn.create_table("people", batch)
            .write_options(stable_row_ids())
            .execute()
            .await
            .unwrap();
        conn
    }

    /// Sources must keep stable row ids; see the create-time gate.
    pub(super) fn stable_row_ids() -> WriteOptions {
        WriteOptions {
            lance_write_params: Some(lance::dataset::WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        }
    }

    #[tokio::test]
    async fn test_create_records_the_definition() {
        let conn = people_db().await;
        let view = conn
            .create_materialized_view("adults", "people")
            .select([("name", "name"), ("shout", "upper(name)")])
            .only_if("age >= 18")
            .limit(10)
            .execute()
            .await
            .unwrap();

        assert_eq!(view.name(), "adults");
        assert_eq!(
            view.definition(),
            &MaterializedViewDefinition {
                source_table: "people".into(),
                projections: vec![
                    ViewProjection {
                        output: "name".into(),
                        expression: "name".into()
                    },
                    ViewProjection {
                        output: "shout".into(),
                        expression: "upper(name)".into()
                    },
                ],
                filter: Some("age >= 18".into()),
                limit: Some(10),
                inputs: vec!["age".into(), "name".into()],
            }
        );

        // The definition round-trips off the stored schema, not the handle.
        let reopened = conn.open_materialized_view("adults").await.unwrap();
        assert_eq!(reopened.definition(), view.definition());
    }

    #[tokio::test]
    async fn test_view_schema_is_derived_from_the_query() {
        let conn = people_db().await;
        let view = conn
            .create_materialized_view("shapes", "people")
            .select([("shout", "upper(name)"), ("next_age", "age + 1")])
            .execute()
            .await
            .unwrap();

        let schema = view.table().schema().await.unwrap();
        assert_eq!(
            schema.field_with_name("shout").unwrap().data_type(),
            &DataType::Utf8
        );
        assert_eq!(
            schema.field_with_name("next_age").unwrap().data_type(),
            &DataType::Int32
        );
        assert_eq!(
            schema
                .field_with_name(SOURCE_ROW_ID_COLUMN)
                .unwrap()
                .data_type(),
            &DataType::UInt64
        );
        assert_eq!(view.table().count_rows(None).await.unwrap(), 0);
    }

    /// No projection selects every source column, expanded now: the schema
    /// captured at creation is the definition.
    #[tokio::test]
    async fn test_default_projection_captures_the_source_schema() {
        let conn = people_db().await;
        let view = conn
            .create_materialized_view("copy", "people")
            .execute()
            .await
            .unwrap();
        assert_eq!(
            view.definition()
                .projections
                .iter()
                .map(|p| p.output.as_str())
                .collect::<Vec<_>>(),
            vec!["name", "age"]
        );
        assert_eq!(view.definition().inputs, vec!["age", "name"]);
    }

    #[tokio::test]
    async fn test_unknown_column_fails_at_create_time() {
        let conn = people_db().await;
        let err = conn
            .create_materialized_view("bad", "people")
            .select([("x", "missing + 1")])
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidExpression { column, .. } if column == "x"));
        assert!(
            !conn
                .table_names()
                .execute()
                .await
                .unwrap()
                .contains(&"bad".to_string())
        );
    }

    #[tokio::test]
    async fn test_unknown_filter_column_fails_at_create_time() {
        let conn = people_db().await;
        let err = conn
            .create_materialized_view("bad", "people")
            .only_if("missing > 1")
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { message } if message.contains("missing")));
    }

    #[tokio::test]
    async fn test_duplicate_output_is_rejected() {
        let conn = people_db().await;
        let err = conn
            .create_materialized_view("bad", "people")
            .select([("dup", "age"), ("dup", "age + 1")])
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::ColumnAlreadyExists { name } if name == "dup"));
    }

    #[tokio::test]
    async fn test_reserved_output_name_is_rejected() {
        let conn = people_db().await;
        let err = conn
            .create_materialized_view("bad", "people")
            .select([(SOURCE_ROW_ID_COLUMN, "age")])
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { message } if message.contains("reserved")));
    }

    #[tokio::test]
    async fn test_missing_source_fails() {
        let conn = connect("memory://").execute().await.unwrap();
        let err = conn
            .create_materialized_view("v", "nope")
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::TableNotFound { .. }));
    }

    /// Provenance has to survive source compactions and updates, and stable
    /// row ids cannot be enabled after a table exists -- so the requirement
    /// is checked at the last moment the caller can still act on it.
    #[tokio::test]
    async fn test_source_without_stable_row_ids_is_refused() {
        let conn = connect("memory://").execute().await.unwrap();
        let batch = record_batch!(("x", Int32, [1, 2])).unwrap();
        conn.create_table("plain", batch).execute().await.unwrap();

        let err = conn
            .create_materialized_view("v", "plain")
            .execute()
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { message } if message.contains("stable row ids"))
        );
        assert!(
            !conn
                .table_names()
                .execute()
                .await
                .unwrap()
                .contains(&"v".to_string())
        );
    }

    #[tokio::test]
    async fn test_name_collision_fails() {
        let conn = people_db().await;
        let err = conn
            .create_materialized_view("people", "people")
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::TableAlreadyExists { .. }));
    }

    #[tokio::test]
    async fn test_a_plain_table_is_not_a_view() {
        let conn = people_db().await;
        let table = conn.open_table("people").execute().await.unwrap();
        let err = MaterializedView::from_table(table).await.unwrap_err();
        assert!(matches!(err, Error::NotAMaterializedView { name } if name == "people"));

        let err = conn.open_materialized_view("people").await.unwrap_err();
        assert!(matches!(err, Error::NotAMaterializedView { .. }));
    }

    /// The reason the kind is tagged: a definition written by a newer version
    /// reads back as a view this one cannot refresh, not as a plain table.
    #[tokio::test]
    async fn test_unrecognized_kind_is_refused_by_name() {
        let conn = people_db().await;
        conn.create_materialized_view("v", "people")
            .execute()
            .await
            .unwrap();
        let table = conn.open_table("v").execute().await.unwrap();
        table
            .as_native()
            .unwrap()
            .replace_schema_metadata(HashMap::from([(
                DEFINITION_META_KEY.to_string(),
                r#"{"kind": "join"}"#.to_string(),
            )]))
            .await
            .unwrap();

        let err = conn.open_materialized_view("v").await.unwrap_err();
        assert!(matches!(err, Error::NotSupported { message } if message.contains("join")));
    }

    #[tokio::test]
    async fn test_list_reports_views_and_only_views() {
        let conn = people_db().await;
        conn.create_materialized_view("adults", "people")
            .only_if("age >= 18")
            .execute()
            .await
            .unwrap();

        let views = conn.list_materialized_views().await.unwrap();
        assert_eq!(
            views.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            vec!["adults"]
        );
        let MaterializedViewKind::Select(definition) = &views[0].kind else {
            panic!("expected a select view");
        };
        assert_eq!(definition.filter.as_deref(), Some("age >= 18"));
    }

    /// A connection override that would create the view without stable row
    /// ids fails loudly instead of committing an unchainable view.
    #[tokio::test]
    async fn test_conflicting_connection_override_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let stable_conn = connect(uri).execute().await.unwrap();
        let batch = record_batch!(("x", Int32, [1])).unwrap();
        stable_conn
            .create_table("src", batch)
            .write_options(stable_row_ids())
            .execute()
            .await
            .unwrap();

        let unstable_conn = connect(uri)
            .storage_options([("new_table_enable_stable_row_ids", "false")])
            .execute()
            .await
            .unwrap();
        let err = unstable_conn
            .create_materialized_view("v", "src")
            .execute()
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { message } if message.contains("stable row ids"))
        );
        assert!(
            !unstable_conn
                .table_names()
                .execute()
                .await
                .unwrap()
                .contains(&"v".to_string())
        );
    }

    /// A committed filter has to be usable as a predicate.
    #[tokio::test]
    async fn test_non_boolean_filter_is_rejected() {
        let conn = people_db().await;
        let err = conn
            .create_materialized_view("bad", "people")
            .only_if("age + 1")
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { message } if message.contains("boolean")),);
    }

    /// Nested references stay dotted paths; resolution is by root field.
    #[tokio::test]
    async fn test_struct_columns_can_be_declared() {
        use arrow_array::{ArrayRef, Int32Array, StructArray};

        let conn = connect("memory://").execute().await.unwrap();
        let ages = StructArray::from(vec![(
            Arc::new(ArrowField::new("age", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![36, 17])) as ArrayRef,
        )]);
        let batch =
            arrow_array::RecordBatch::try_from_iter(vec![("metadata", Arc::new(ages) as ArrayRef)])
                .unwrap();
        conn.create_table("people", batch)
            .write_options(stable_row_ids())
            .execute()
            .await
            .unwrap();

        let view = conn
            .create_materialized_view("ages", "people")
            .select([("age", "metadata.age")])
            .only_if("metadata.age >= 18")
            .execute()
            .await
            .unwrap();
        assert_eq!(view.definition().inputs, vec!["metadata.age"]);
        let schema = view.table().schema().await.unwrap();
        assert_eq!(
            schema.field_with_name("age").unwrap().data_type(),
            &DataType::Int32
        );
    }

    /// A newer-kind view must not disappear from the listing.
    #[tokio::test]
    async fn test_unrecognized_kind_is_listed_with_its_kind() {
        let conn = people_db().await;
        conn.create_materialized_view("v", "people")
            .execute()
            .await
            .unwrap();
        let table = conn.open_table("v").execute().await.unwrap();
        table
            .as_native()
            .unwrap()
            .replace_schema_metadata(HashMap::from([(
                DEFINITION_META_KEY.to_string(),
                r#"{"kind": "join"}"#.to_string(),
            )]))
            .await
            .unwrap();

        let views = conn.list_materialized_views().await.unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].name, "v");
        assert_eq!(
            views[0].kind,
            MaterializedViewKind::Unrecognized {
                kind: "join".into()
            }
        );
    }

    /// Remote connections are refused before any request is made.
    #[cfg(feature = "remote")]
    #[tokio::test]
    async fn test_remote_connection_is_refused_up_front() {
        let conn = connect("db://nowhere")
            .api_key("sk_test")
            .region("us-east-1")
            .execute()
            .await
            .unwrap();
        let err = conn
            .create_materialized_view("v", "src")
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }));
        let err = conn.open_materialized_view("v").await.unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }));
        let err = conn.list_materialized_views().await.unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }));
    }

    /// A definition must evaluate identically across refreshes; anything
    /// less makes incremental maintenance a mix of evaluations.
    #[tokio::test]
    async fn test_volatile_and_unstable_expressions_are_rejected() {
        let conn = people_db().await;
        for expression in [
            "random()",
            "now()",
            "version()",
            "arrow_typeof(age)",
            "arrow_metadata(age, 'k')",
        ] {
            let err = conn
                .create_materialized_view("bad", "people")
                .select([("x", expression)])
                .execute()
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::InvalidExpression { message, .. }
                    if message.contains("not immutable")),
                "{expression} was not rejected"
            );
        }
        for filter in ["age > random() * 100", "age >= 0 and now() is not null"] {
            let err = conn
                .create_materialized_view("bad", "people")
                .only_if(filter)
                .execute()
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::InvalidInput { message } if message.contains("not immutable")),
                "{filter} was not rejected"
            );
        }
    }

    #[tokio::test]
    async fn test_drop_is_drop_table() {
        let conn = people_db().await;
        conn.create_materialized_view("v", "people")
            .execute()
            .await
            .unwrap();
        conn.drop_table("v", &[]).await.unwrap();
        assert!(conn.list_materialized_views().await.unwrap().is_empty());
    }
}
