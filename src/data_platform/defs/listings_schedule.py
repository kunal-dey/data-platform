import dagster as dg

# Keep in sync with data_extraction/utils/screener_fetch.TABLE_NAMES
_SCREENER_TABLES = [
    "quarterly_results",
    "profit_loss",
    "balance_sheet",
    "cash_flow",
    "ratios",
    "shareholding_quarterly",
    "shareholding_yearly",
]
_SCREENER_KEYS = [["bronze_screener", name] for name in _SCREENER_TABLES]

listings_job = dg.define_asset_job(
    name="listings_job",
    selection=dg.AssetSelection.keys(["bronze_listings", "equity_universe"]),
    description="Load NSE/BSE equity listings into bronze_listings.equity_universe",
)

screener_job = dg.define_asset_job(
    name="screener_job",
    selection=dg.AssetSelection.keys(*_SCREENER_KEYS),
    description="Load Screener period tables into bronze_screener.*",
)


@dg.schedule(
    name="listings_weekly",
    cron_schedule="0 23 * * 0",  # Sunday 23:00 IST
    job=listings_job,
    execution_timezone="Asia/Kolkata",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def listings_weekly_schedule():
    return dg.RunRequest()


@dg.schedule(
    name="screener_weekdays",
    cron_schedule="0 18 * * 1-5",  # Mon–Fri 18:00 IST
    job=screener_job,
    execution_timezone="Asia/Kolkata",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def screener_weekdays_schedule():
    return dg.RunRequest()


@dg.definitions
def extraction_schedules():
    return dg.Definitions(
        jobs=[listings_job, screener_job],
        schedules=[listings_weekly_schedule, screener_weekdays_schedule],
    )
