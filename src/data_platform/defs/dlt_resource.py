import dagster as dg
from dagster_dlt import DagsterDltResource


@dg.definitions
def dlt_resource():
    return dg.Definitions(resources={"dlt": DagsterDltResource()})
