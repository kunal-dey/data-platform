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
    # Async scrape + Iceberg are memory-heavy; child crashes often = OOM under
    # default multiprocess concurrency.
    executor_def=dg.multiprocess_executor.configured({"max_concurrent": 1}),
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


@dg.schedule(
    name="stock_news_weekdays",
    cron_schedule="30 7 * * 1-5",  # Mon–Fri 07:30 IST
    job=stock_news_job,
    execution_timezone="Asia/Kolkata",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def stock_news_weekdays_schedule():
    return dg.RunRequest()


@dg.definitions
def extraction_schedules():
    return dg.Definitions(
        jobs=[listings_job, screener_job, stock_news_job],
        schedules=[
            listings_weekly_schedule,
            screener_weekdays_schedule,
            stock_news_weekdays_schedule,
        ],
    )
