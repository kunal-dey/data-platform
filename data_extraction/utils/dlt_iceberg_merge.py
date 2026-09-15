"""Work around dlt Iceberg merge re-adding dropped Screener columns via union_by_name."""

from __future__ import annotations

import dlt.common.libs.pyiceberg as _pi
from dlt.common.libs.pyiceberg import (
    ensure_iceberg_compatible_arrow_data,
    get_columns_names_with_prop,
    get_first_column_name_with_prop,
)
from dlt.common.schema.typing import TTableSchema

_orig_merge = _pi.merge_iceberg_table
_patched = False


def patch_iceberg_merge_for_screener() -> None:
    global _patched
    if _patched:
        return

    def merge_iceberg_table(
        table,
        data,
        schema: TTableSchema,
        load_table_name: str,
    ) -> None:
        strategy = schema["x-merge-strategy"]  # type: ignore[typeddict-item]
        if strategy not in ("upsert", "insert-only"):
            return _orig_merge(table, data, schema, load_table_name)

        if hasattr(table, "refresh"):
            table.refresh()

        if "parent" in schema:
            join_cols = [get_first_column_name_with_prop(schema, "unique")]
        else:
            join_cols = get_columns_names_with_prop(schema, "primary_key")

        import pyarrow as pa

        arrow = data.to_table() if hasattr(data, "to_table") else data
        for rb in arrow.to_batches(max_chunksize=1_000):
            batch_tbl = ensure_iceberg_compatible_arrow_data(
                pa.Table.from_batches([rb])
            )
            target_arrow = table.schema().as_arrow()
            for field in target_arrow:
                if field.name not in batch_tbl.column_names:
                    batch_tbl = batch_tbl.append_column(
                        field.name,
                        pa.nulls(batch_tbl.num_rows, type=field.type),
                    )
            batch_tbl = batch_tbl.select([f.name for f in target_arrow])
            table.upsert(
                df=batch_tbl,
                join_cols=join_cols,
                when_matched_update_all=strategy == "upsert",
                when_not_matched_insert_all=True,
                case_sensitive=True,
            )

    _pi.merge_iceberg_table = merge_iceberg_table
    _patched = True
