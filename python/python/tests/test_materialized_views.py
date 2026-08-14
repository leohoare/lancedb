# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import lancedb
import pytest
from lancedb.materialized_view import MaterializedViewDefinition


STABLE_ROW_IDS = {"new_table_enable_stable_row_ids": "true"}


def make_db(tmp_path):
    db = lancedb.connect(tmp_path, storage_options=STABLE_ROW_IDS)
    db.create_table(
        "people",
        [
            {"name": "ada", "age": 36},
            {"name": "kid", "age": 7},
            {"name": "grace", "age": 85},
        ],
    )
    return db


def test_create_refresh_and_query(tmp_path):
    db = make_db(tmp_path)
    view = db.create_materialized_view(
        "adults",
        "people",
        select=["name", ("shout", "upper(name)")],
        where="age >= 18",
    )
    assert view.name == "adults"
    assert view.table.count_rows() == 0

    result = view.refresh()
    assert result.mode == "rebuild"
    assert result.rows_written == 2

    rows = view.table.search().to_list()
    assert sorted(row["shout"] for row in rows) == ["ADA", "GRACE"]


def test_definition_round_trips(tmp_path):
    db = make_db(tmp_path)
    db.create_materialized_view("adults", "people", where="age >= 18")

    view = db.open_materialized_view("adults")
    assert view.definition == MaterializedViewDefinition(
        source_table="people",
        projections=[("name", "`name`"), ("age", "`age`")],
        filter="age >= 18",
        inputs=["age", "name"],
    )


def test_incremental_refresh_after_append(tmp_path):
    db = make_db(tmp_path)
    view = db.create_materialized_view("copy", "people")
    view.refresh()

    db.open_table("people").add([{"name": "alan", "age": 41}])
    result = view.refresh()
    assert result.mode == "incremental"
    assert result.rows_written == 1
    assert view.table.count_rows() == 4

    assert view.refresh().mode == "no_op"


def test_full_refresh_after_update(tmp_path):
    db = make_db(tmp_path)
    view = db.create_materialized_view("copy", "people")
    view.refresh()

    db.open_table("people").update(where="name = 'kid'", values={"age": 8})
    result = view.refresh()
    assert result.mode == "rebuild"
    rows = view.table.search().to_list()
    assert sorted(row["age"] for row in rows) == [8, 36, 85]


def test_list_and_not_a_view(tmp_path):
    db = make_db(tmp_path)
    db.create_materialized_view("adults", "people", where="age >= 18")

    assert db.list_materialized_views() == ["adults"]
    with pytest.raises(ValueError, match="not a materialized view"):
        db.open_materialized_view("people")


def test_invalid_expression_fails_at_create(tmp_path):
    db = make_db(tmp_path)
    with pytest.raises(Exception, match="missing"):
        db.create_materialized_view("bad", "people", select=[("x", "missing + 1")])
    assert "bad" not in db.list_tables().tables


@pytest.mark.asyncio
async def test_async_create_refresh_and_open(tmp_path):
    db = await lancedb.connect_async(tmp_path, storage_options=STABLE_ROW_IDS)
    await db.create_table("people", [{"name": "ada", "age": 36}])

    view = await db.create_materialized_view(
        "shouts", "people", select=[("shout", "upper(name)")]
    )
    result = await view.refresh()
    assert result.mode == "rebuild"
    assert result.rows_written == 1

    reopened = await db.open_materialized_view("shouts")
    definition = await reopened.definition()
    assert definition.projections == [("shout", "upper(name)")]
    assert await db.list_materialized_views() == ["shouts"]


@pytest.mark.asyncio
async def test_async_incremental(tmp_path):
    db = await lancedb.connect_async(tmp_path, storage_options=STABLE_ROW_IDS)
    await db.create_table("people", [{"name": "ada", "age": 36}])
    view = await db.create_materialized_view("copy", "people")
    await view.refresh()

    table = await db.open_table("people")
    await table.add([{"name": "alan", "age": 41}])
    result = await view.refresh()
    assert result.mode == "incremental"
    assert result.rows_written == 1


def test_source_requires_stable_row_ids(tmp_path):
    db = lancedb.connect(tmp_path)
    db.create_table("plain", [{"x": 1}])
    with pytest.raises(Exception, match="stable row ids"):
        db.create_materialized_view("v", "plain")


def test_bare_select_names_are_quoted(tmp_path):
    db = lancedb.connect(tmp_path, storage_options=STABLE_ROW_IDS)
    db.create_table("odd_names", [{"order item": "widget", "select": 2}])

    view = db.create_materialized_view(
        "quoted", "odd_names", select=["order item", "select"]
    )
    result = view.refresh()
    assert result.rows_written == 1
    rows = view.table.search().to_list()
    assert rows[0]["order item"] == "widget"
    assert rows[0]["select"] == 2


@pytest.mark.asyncio
async def test_async_remote_is_refused_without_network():
    db = await lancedb.connect_async(
        "db://nowhere", api_key="sk_test", region="us-east-1"
    )
    with pytest.raises(NotImplementedError, match="local"):
        await db.create_materialized_view("v", "src")
    with pytest.raises(NotImplementedError, match="local"):
        await db.open_materialized_view("v")
    with pytest.raises(NotImplementedError, match="local"):
        await db.list_materialized_views()
