import dagster as dg

listings_job = dg.define_asset_job(
    name="listings_job",
    selection=dg.AssetSelection.keys(["bronze_listings", "equity_universe"]),
    description="Load NSE/BSE equity listings into bronze_listings.equity_universe",
)


@dg.schedule(
    name="listings_weekly",
    cron_schedule="0 23 * * 0",  # Sunday 23:00
    job=listings_job,
    execution_timezone="Asia/Kolkata",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def listings_weekly_schedule():
    return dg.RunRequest()


@dg.definitions
def listings_schedules():
    return dg.Definitions(
        jobs=[listings_job],
        schedules=[listings_weekly_schedule],
    )
