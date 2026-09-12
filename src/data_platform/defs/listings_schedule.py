import dagster as dg

listings_job = dg.define_asset_job(
    name="listings_job",
    selection=dg.AssetSelection.key_prefixes(["bronze_listings"]),
    description="Load NSE/BSE equity listings into bronze_listings.equity_universe",
)

screener_job = dg.define_asset_job(
    name="screener_job",
    selection=dg.AssetSelection.key_prefixes(["bronze_screener"]),
    description="Load Screener period tables into bronze_screener.*",
    # Full-universe scrape can run many hours on small EC2.
    tags={"dagster/max_runtime": 86400},
)

stock_news_job = dg.define_asset_job(
    name="stock_news_job",
    selection=dg.AssetSelection.key_prefixes(["bronze_economic_times"]),
    description="Scrape ET companies/articles into bronze_economic_times.*",
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
    name="screener_daily",
    cron_schedule="15 0 * * *",  # Every day 00:15 IST
    job=screener_job,
    execution_timezone="Asia/Kolkata",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def screener_daily_schedule():
    return dg.RunRequest(
        tags={"dagster/concurrency_key": "screener_job"},
    )


@dg.schedule(
    name="stock_news_daily",
    cron_schedule="30 9 * * *",  # Every day 09:30 IST
    job=stock_news_job,
    execution_timezone="Asia/Kolkata",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def stock_news_daily_schedule():
    return dg.RunRequest()


@dg.definitions
def extraction_schedules():
    return dg.Definitions(
        jobs=[listings_job, screener_job, stock_news_job],
        schedules=[
            listings_weekly_schedule,
            screener_daily_schedule,
            stock_news_daily_schedule,
        ],
    )
