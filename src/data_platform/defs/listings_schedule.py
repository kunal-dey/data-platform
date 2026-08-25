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

# companies/articles first; conform_articles deps ensure it runs after articles
_STOCK_NEWS_KEYS = [
    ["bronze_economic_times", "companies"],
    ["bronze_economic_times", "articles"],
    ["bronze_economic_times", "conform_articles"],
]

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

stock_news_job = dg.define_asset_job(
    name="stock_news_job",
    selection=dg.AssetSelection.keys(*_STOCK_NEWS_KEYS),
    description=(
        "Scrape ET companies/articles, then LLM-conform new articles "
        "(≤7 days) into bronze_economic_times.conform_articles"
    ),
    # Avoid multiprocess children: on small EC2 (~1GB) the kernel OOM-kills
    # them (SIGKILL / -9), which Dagster surfaces as ChildProcessCrashException.
    executor_def=dg.in_process_executor,
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
    return dg.RunRequest()


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
