// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

import { RefreshMaterializedViewResult } from "./native";
import { Table } from "./table";

/** Schema metadata key holding a materialized view's definition. */
export const DEFINITION_META_KEY = "mv.definition";

/** The query that defines a materialized view. */
export interface MaterializedViewDefinition {
  /** Name of the source table, in the same database as the view. */
  sourceTable: string;
  /** `[output column, SQL expression]` pairs, in view schema order. */
  projections: [string, string][];
  /** SQL predicate selecting the source rows the view holds. */
  filter?: string;
  /** Cap on the number of rows the view holds. */
  limit?: number;
  /** Source columns the projections and filter read. */
  inputs: string[];
}

/**
 * The view's columns: column names, `[alias, SQL expression]` pairs, or a
 * record of the same. A bare name projects itself.
 */
export type MaterializedViewSelect =
  | (string | [string, string])[]
  | Record<string, string>;

/** @internal Normalize a select argument into `[alias, expression]` pairs. */
export function normalizeSelect(
  select?: MaterializedViewSelect,
): [string, string][] | undefined {
  if (select === undefined) {
    return undefined;
  }
  if (Array.isArray(select)) {
    return select.map((item) =>
      typeof item === "string" ? [item, item] : item,
    );
  }
  return Object.entries(select);
}

/** @internal Parse a definition off a table's stored schema metadata. */
export function definitionFromMetadata(
  metadata: Map<string, string>,
  name: string,
): MaterializedViewDefinition {
  const raw = metadata.get(DEFINITION_META_KEY);
  if (raw === undefined) {
    throw new Error(`Table '${name}' is not a materialized view`);
  }
  // biome-ignore lint/suspicious/noExplicitAny: raw JSON
  const value: any = JSON.parse(raw);
  if (value.kind !== "select") {
    throw new Error(
      `materialized view '${name}' is defined by '${value.kind}', which this ` +
        "version of lancedb cannot refresh",
    );
  }
  return {
    sourceTable: value.source_table,
    // biome-ignore lint/suspicious/noExplicitAny: raw JSON
    projections: (value.projections ?? []).map((p: any) => [
      p.output,
      p.expression,
    ]),
    filter: value.filter ?? undefined,
    limit: value.limit ?? undefined,
    inputs: value.inputs ?? [],
  };
}

/**
 * A handle on a materialized view: its table plus its definition.
 *
 * Obtained from {@link Connection#createMaterializedView} or
 * {@link Connection#openMaterializedView}. The view is a normal table --
 * queries, indexes and search all apply through {@link MaterializedView#table}
 * -- whose contents are maintained by {@link MaterializedView#refresh}.
 */
export class MaterializedView {
  private readonly inner: Table;

  constructor(table: Table) {
    this.inner = table;
  }

  get name(): string {
    return this.inner.name;
  }

  /** The view, as the table it is. */
  table(): Table {
    return this.inner;
  }

  /** The query that defines the view, read from its stored schema. */
  async definition(): Promise<MaterializedViewDefinition> {
    const schema = await this.inner.schema();
    return definitionFromMetadata(schema.metadata, this.name);
  }

  /**
   * Recompute the view from its source.
   *
   * The refresh is incremental when the source only gained rows since the
   * last one, and otherwise rebuilds. `full` forces a rebuild;
   * `sourceVersion` refreshes to that source version instead of the latest.
   */
  async refresh(options?: {
    full?: boolean;
    sourceVersion?: number;
  }): Promise<RefreshMaterializedViewResult> {
    return await this.inner.refreshMaterializedView(
      options?.full,
      options?.sourceVersion,
    );
  }
}
