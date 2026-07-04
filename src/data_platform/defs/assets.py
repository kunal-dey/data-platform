import dagster as dg


@dg.asset
def test_asset() -> str:
    return "test asset materialized successfully"
