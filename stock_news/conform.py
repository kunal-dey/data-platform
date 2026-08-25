from __future__ import annotations

"""LLM company→ticker match for bronze ET articles.

Writes ``bronze_economic_times.conform_articles`` (Iceberg) via dlt merge upsert.

Pending set = articles from the last ``MAX_ARTICLE_AGE_DAYS`` days whose ``url``
is not already in ``conform_articles`` (incremental — same idea as merge upsert).
"""

import asyncio
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from stock_news.symbol_resolve import clear_resolver_cache, get_resolver

log = logging.getLogger(__name__)

SCHEMA_ET = "bronze_economic_times"
ARTICLES_FQN = f"{SCHEMA_ET}.articles"
CONFORM_FQN = f"{SCHEMA_ET}.conform_articles"
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"
DEFAULT_BATCH_SIZE = 8
DEFAULT_CONCURRENCY = 2
CONTENT_CHARS = 1800
CANDIDATES_PER_ARTICLE = 25
MAX_ARTICLE_AGE_DAYS = 7


def _run_coro(coro):
    """Run a coroutine; safe if a loop is already running (some Dagster executors)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _ensure_data_extraction_path() -> None:
    de = Path(__file__).resolve().parents[1] / "data_extraction"
    if str(de) not in sys.path:
        sys.path.insert(0, str(de))


def _parse_et_dates(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(" IST", "", regex=False)
        .str.replace(",", " ", regex=False)
        .str.replace(r"(?i)(?<=\d)(AM|PM)", r" \1", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    parsed = pd.to_datetime(cleaned, format="%d %b %Y %I:%M %p", errors="coerce")
    miss = parsed.isna() & cleaned.ne("") & cleaned.ne("nan")
    if miss.any():
        parsed.loc[miss] = pd.to_datetime(
            cleaned.loc[miss], format="mixed", dayfirst=True, errors="coerce"
        )
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(
            "Asia/Kolkata", ambiguous="NaT", nonexistent="NaT"
        )
    return parsed


def _read_iceberg_table(fqn: str, fields: tuple[str, ...] | None = None) -> pd.DataFrame:
    _ensure_data_extraction_path()
    from utils.dlt_lake_config import load_glue_catalog

    catalog = load_glue_catalog()
    try:
        table = catalog.load_table(fqn)
    except Exception as e:
        log.info("Iceberg table %s not available: %s", fqn, e)
        return pd.DataFrame()
    scan = table.scan(selected_fields=fields) if fields else table.scan()
    return scan.to_arrow().to_pandas()


def _existing_conform_urls() -> set[str]:
    df = _read_iceberg_table(CONFORM_FQN, fields=("url",))
    if df.empty or "url" not in df.columns:
        return set()
    return {str(u).strip() for u in df["url"] if u and str(u).strip()}


def pending_articles(limit: int | None = None) -> pd.DataFrame:
    """Last ``MAX_ARTICLE_AGE_DAYS`` days of articles not yet in conform_articles."""
    df = _read_iceberg_table(
        ARTICLES_FQN,
        fields=("url", "heading", "content", "date", "companies"),
    )
    if df.empty:
        return df

    df = df.copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].ne("") & df["url"].ne("nan")]
    df["_parsed"] = _parse_et_dates(df["date"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_ARTICLE_AGE_DAYS)
    parsed_utc = df["_parsed"].dt.tz_convert("UTC")
    df = df[parsed_utc.notna() & (parsed_utc >= cutoff)]

    done = _existing_conform_urls()
    if done:
        df = df[~df["url"].isin(done)]

    df = df.sort_values("_parsed", ascending=False).drop(columns=["_parsed"])
    if limit is not None:
        df = df.head(int(limit))
    return df.reset_index(drop=True)


SYSTEM_PROMPT = """\
You are a financial news analyst for Indian equity markets.

You receive a BATCH of Economic Times articles. For EACH article you get:
1) Article metadata + body
2) A CANDIDATE LIST of listed companies (symbol | exchange | company_name)

Your ONLY job for every article:
- Identify which listed stocks the article is actually about
- MATCH names as written in the article to the correct SYMBOL from that
  article's candidate list (think carefully — prefer true subjects, not
  weak name collisions)

Rules:
- Return exactly one result object per input article (same url)
- ONLY use symbols that appear in that article's candidate list
- If no candidate fits, return an empty stock_matches list
- Do not invent tickers
- Prefer NSE when both NSE and BSE appear for the same symbol
- Do not rewrite headlines, summarize, score sentiment, or invent topics
"""


class StockMatchOut(BaseModel):
    company_as_in_article: str = Field(
        description="Company/stock name as referred to in the article"
    )
    symbol: str = Field(description="Ticker from the candidate list only")
    exchange: str = Field(description="NSE or BSE from the candidate list")
    listed_company_name: str = Field(
        description="Official company_name from the candidate list"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How sure you are this article is about this listing"
    )


class ConformArticleOut(BaseModel):
    url: str = Field(description="Exact article URL from the input batch")
    stock_matches: list[StockMatchOut] = Field(
        default_factory=list,
        description="Article companies matched to listing symbols from candidates",
    )


class ConformBatchOut(BaseModel):
    articles: list[ConformArticleOut] = Field(
        description="One match result per input article in the batch"
    )


def _as_list(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [p.strip() for p in text.split("|") if p.strip()]


def _format_candidates(hits: list[Any]) -> str:
    if not hits:
        return "(no candidates)"
    lines = ["symbol | exchange | company_name"]
    for h in hits:
        lines.append(f"{h.symbol} | {h.exchange} | {h.company_name}")
    return "\n".join(lines)


def _chunk_df(df: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    size = max(1, int(size))
    return [df.iloc[i : i + size].copy() for i in range(0, len(df), size)]


def _validate_llm_matches(
    matches: list[StockMatchOut],
    *,
    allowed: set[tuple[str, str]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    resolver = get_resolver()
    companies: list[str] = []
    symbols: list[str] = []
    detail: list[dict[str, Any]] = []
    seen_co: set[str] = set()
    seen_sym: set[str] = set()

    for m in matches:
        symbol = m.symbol.strip().upper()
        exchange = m.exchange.strip().upper()
        if not symbol or not resolver.has_symbol(symbol):
            continue
        if (exchange, symbol) not in allowed and not any(
            s == symbol for _, s in allowed
        ):
            continue
        if exchange not in {"NSE", "BSE"}:
            exchange = "NSE" if ("NSE", symbol) in allowed else exchange

        co = m.company_as_in_article.strip()
        if co and co.lower() not in seen_co:
            seen_co.add(co.lower())
            companies.append(co)
        if symbol not in seen_sym:
            seen_sym.add(symbol)
            symbols.append(symbol)
        detail.append(
            {
                "company_as_in_article": co,
                "symbol": symbol,
                "exchange": exchange,
                "listed_company_name": m.listed_company_name.strip(),
                "confidence": m.confidence,
                "matched_by": "llm",
            }
        )
    return companies, symbols, detail


def _fallback_from_source(
    source_cos: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    resolver = get_resolver()
    companies: list[str] = []
    symbols: list[str] = []
    matches: list[dict[str, Any]] = []
    for name in source_cos:
        hits = resolver.resolve_name(name, max_hits=1)
        if not hits or hits[0].score < 0.85:
            continue
        h = hits[0]
        if name not in companies:
            companies.append(name)
        if h.symbol not in symbols:
            symbols.append(h.symbol)
        matches.append(
            {
                "company_as_in_article": name,
                "symbol": h.symbol,
                "exchange": h.exchange,
                "listed_company_name": h.company_name,
                "confidence": "medium",
                "matched_by": "fallback_listings",
            }
        )
    return companies, symbols, matches


def _apply_listings_guardrail(
    source_cos: list[str],
    companies: list[str],
    symbols: list[str],
    matches: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    if not source_cos:
        return companies, symbols, matches

    resolver = get_resolver()
    seen_co = {c.lower() for c in companies}
    seen_sym = set(symbols)
    out_matches = list(matches)

    for name in source_cos:
        hits = resolver.resolve_name(name, max_hits=1)
        if not hits or hits[0].score < 0.85:
            continue
        h = hits[0]
        if h.symbol in seen_sym:
            if name and name.lower() not in seen_co:
                seen_co.add(name.lower())
                companies.append(name)
            continue
        seen_sym.add(h.symbol)
        symbols.append(h.symbol)
        if name and name.lower() not in seen_co:
            seen_co.add(name.lower())
            companies.append(name)
        out_matches.append(
            {
                "company_as_in_article": name,
                "symbol": h.symbol,
                "exchange": h.exchange,
                "listed_company_name": h.company_name,
                "confidence": "high" if h.score >= 0.95 else "medium",
                "matched_by": "listings_guardrail",
            }
        )
    return companies, symbols, out_matches


def _prepare_batch_context(
    batch: pd.DataFrame,
) -> tuple[str, dict[str, dict[str, Any]]]:
    resolver = get_resolver()
    meta: dict[str, dict[str, Any]] = {}
    parts: list[str] = [
        f"Process this batch of {len(batch)} articles. "
        "For each article return only stock_matches (same url). "
        "Do not rewrite headlines or write summaries.\n"
    ]

    for i, (_, row) in enumerate(batch.iterrows(), start=1):
        url = str(row["url"])
        source_cos = _as_list(row.get("companies"))
        candidates = resolver.candidates_for_article(
            headline=str(row.get("heading") or ""),
            content=str(row.get("content") or ""),
            source_companies=source_cos,
            limit=CANDIDATES_PER_ARTICLE,
        )
        allowed = {(h.exchange, h.symbol) for h in candidates}
        meta[url] = {
            "source_companies": source_cos,
            "allowed": allowed,
            "source_date": str(row.get("date") or ""),
            "source_heading": str(row.get("heading") or ""),
        }
        content = str(row.get("content") or "")[:CONTENT_CHARS]
        parts.append(
            f"===== ARTICLE {i}/{len(batch)} =====\n"
            f"URL: {url}\n"
            f"Headline: {row.get('heading') or ''}\n"
            f"Date: {row.get('date') or ''}\n"
            f"Linked companies from scrape: "
            f"{', '.join(source_cos) if source_cos else '(none)'}\n\n"
            f"CANDIDATE LIST for this article "
            f"(choose symbols ONLY from here):\n"
            f"{_format_candidates(candidates)}\n\n"
            f"Article body:\n{content}\n"
        )

    return "\n".join(parts), meta


def _row_from_matches(
    *,
    url: str,
    info: dict[str, Any],
    companies: list[str],
    symbols: list[str],
    matches: list[dict[str, Any]],
    stamp: datetime,
) -> dict[str, Any]:
    return {
        "url": url,
        "companies_mentioned": companies,
        "symbols": symbols,
        "symbol_matches": matches,
        "source_heading": info["source_heading"],
        "source_date": info["source_date"],
        "source_companies": info["source_companies"],
        "model_name": MODEL_NAME,
        "conformed_at": stamp,
    }


def _materialize_batch_result(
    data: ConformBatchOut,
    *,
    meta: dict[str, dict[str, Any]],
    stamp: datetime,
) -> list[dict[str, Any]]:
    by_url = {a.url.strip(): a for a in data.articles}
    out: list[dict[str, Any]] = []

    for url, info in meta.items():
        article = by_url.get(url)
        source_cos = info["source_companies"]
        allowed = info["allowed"]

        if article is None:
            log.warning("LLM batch missing url=%s — using listings guardrail only", url)
            companies, symbols, matches = _fallback_from_source(source_cos)
            out.append(
                _row_from_matches(
                    url=url,
                    info=info,
                    companies=companies,
                    symbols=symbols,
                    matches=matches,
                    stamp=stamp,
                )
            )
            continue

        companies, symbols, matches = _validate_llm_matches(
            article.stock_matches, allowed=allowed
        )
        companies, symbols, matches = _apply_listings_guardrail(
            source_cos, companies, symbols, matches
        )
        out.append(
            _row_from_matches(
                url=url,
                info=info,
                companies=companies,
                symbols=symbols,
                matches=matches,
                stamp=stamp,
            )
        )
    return out


async def _run_agent_batches(
    df: pd.DataFrame,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict[str, Any]]:
    from dotenv import load_dotenv
    from pydantic_ai import Agent, PromptedOutput
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from services import VLLM_API_KEY
    from services.gpu_operator import GpuOperator
    from services.models import GpuPodConfig

    load_dotenv()
    clear_resolver_cache()
    get_resolver()

    batches = _chunk_df(df, batch_size)
    sem = asyncio.Semaphore(max(1, concurrency))
    out: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc)

    log.info(
        "LLM batching: %s articles -> %s batches (size=%s, parallel=%s)",
        len(df),
        len(batches),
        batch_size,
        concurrency,
    )

    with GpuOperator(config=GpuPodConfig(model_type="llm")) as ops:
        provider = OpenAIProvider(base_url=ops.llm_url, api_key=VLLM_API_KEY)
        model = OpenAIChatModel(MODEL_NAME, provider=provider)
        agent = Agent(
            model=model,
            output_type=PromptedOutput(ConformBatchOut),
            system_prompt=SYSTEM_PROMPT,
        )

        async def run_one_batch(
            batch_idx: int, batch: pd.DataFrame
        ) -> list[dict[str, Any]]:
            async with sem:
                prompt, meta = _prepare_batch_context(batch)
                try:
                    result = await agent.run(prompt)
                    data = result.output
                    if not isinstance(data, ConformBatchOut):
                        data = ConformBatchOut.model_validate(data)
                    rows = _materialize_batch_result(data, meta=meta, stamp=stamp)
                    log.info(
                        "Batch %s/%s done (%s articles)",
                        batch_idx,
                        len(batches),
                        len(rows),
                    )
                    return rows
                except Exception:
                    log.exception(
                        "Batch %s/%s failed (%s urls)",
                        batch_idx,
                        len(batches),
                        len(batch),
                    )
                    fallback: list[dict[str, Any]] = []
                    for url, info in meta.items():
                        cos, syms, matches = _fallback_from_source(
                            info["source_companies"]
                        )
                        fallback.append(
                            _row_from_matches(
                                url=url,
                                info=info,
                                companies=cos,
                                symbols=syms,
                                matches=matches,
                                stamp=stamp,
                            )
                        )
                    return fallback

        tasks = [
            run_one_batch(i, batch) for i, batch in enumerate(batches, start=1)
        ]
        for coro in asyncio.as_completed(tasks):
            out.extend(await coro)

    return out


def run_conform_rows(
    *,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Return conform rows for pending (≤7d, not yet conformed) articles."""
    pending = pending_articles(limit=limit)
    if pending.empty:
        log.info(
            "No pending articles to conform (≤%s days, not already conformed)",
            MAX_ARTICLE_AGE_DAYS,
        )
        return []

    log.info(
        "Conforming %s new articles (≤%s days old) via RunPod (%s)",
        len(pending),
        MAX_ARTICLE_AGE_DAYS,
        MODEL_NAME,
    )
    return _run_coro(
        _run_agent_batches(
            pending, batch_size=batch_size, concurrency=concurrency
        )
    )


def conform_articles(
    *,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """CLI helper: run LLM conform and load to Iceberg via dlt."""
    rows = run_conform_rows(
        limit=limit, batch_size=batch_size, concurrency=concurrency
    )
    if not rows:
        return 0

    _ensure_data_extraction_path()
    import dlt
    from utils.dlt_lake_config import filesystem_destination

    @dlt.resource(
        name="conform_articles",
        primary_key="url",
        write_disposition={"disposition": "merge", "strategy": "upsert"},
        table_format="iceberg",
    )
    def _res():
        yield from rows

    info = dlt.pipeline(
        pipeline_name="stock_news_conform",
        destination=filesystem_destination(),
        dataset_name=SCHEMA_ET,
    ).run(_res())
    log.info("Wrote %s conformed articles -> %s (%s)", len(rows), CONFORM_FQN, info)
    return len(rows)
