"""Resolve company / stock names → exchange symbols via bronze_listings (Iceberg)."""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

_SUFFIX_RE = re.compile(
    r"\b("
    r"limited|ltd\.?|private|pvt\.?|plc|inc\.?|corp\.?|corporation|"
    r"company|co\.?|the"
    r")\b",
    re.I,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_LETTER_DIGIT_RE = re.compile(r"([a-z])(\d)", re.I)
_DIGIT_LETTER_RE = re.compile(r"(\d)([a-z])", re.I)


def normalize_name(name: str) -> str:
    text = str(name or "").lower().strip()
    text = _SUFFIX_RE.sub(" ", text)
    text = _LETTER_DIGIT_RE.sub(r"\1 \2", text)
    text = _DIGIT_LETTER_RE.sub(r"\1 \2", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


class SymbolHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    exchange: str
    company_name: str
    score: float


def _load_equity_universe_df() -> pd.DataFrame:
    """Read bronze_listings.equity_universe from Glue/Iceberg."""
    root = Path(__file__).resolve().parents[1]
    de = root / "data_extraction"
    if str(de) not in sys.path:
        sys.path.insert(0, str(de))

    from utils.dlt_lake_config import load_glue_catalog

    catalog = load_glue_catalog()
    table = catalog.load_table("bronze_listings.equity_universe")
    arrow = table.scan(
        selected_fields=("exchange", "symbol", "company_name")
    ).to_arrow()
    df = arrow.to_pandas()
    if "is_active" in df.columns:
        df = df[df["is_active"].fillna(True).astype(bool)]
    return df


class SymbolResolver:
    """Exact + fuzzy name match against equity universe."""

    def __init__(self, universe: pd.DataFrame) -> None:
        self._by_norm: dict[str, list[tuple[str, str, str]]] = {}
        self._by_symbol: set[str] = set()
        for _, r in universe.iterrows():
            symbol = str(r["symbol"]).strip().upper()
            exchange = str(r["exchange"]).strip().upper()
            cname = str(r["company_name"] or "").strip()
            if not symbol or not cname:
                continue
            self._by_symbol.add(symbol)
            key = normalize_name(cname)
            if not key:
                continue
            self._by_norm.setdefault(key, []).append((symbol, exchange, cname))

        for key, rows in self._by_norm.items():
            rows.sort(key=lambda x: (0 if x[1] == "NSE" else 1, x[0]))

    @classmethod
    def from_iceberg(cls) -> SymbolResolver:
        return cls(_load_equity_universe_df())

    # Back-compat alias
    from_warehouse = from_iceberg

    def resolve_name(self, name: str, *, max_hits: int = 3) -> list[SymbolHit]:
        key = normalize_name(name)
        if not key:
            return []

        hits: list[SymbolHit] = []
        seen: set[tuple[str, str]] = set()

        def add(symbol: str, exchange: str, cname: str, score: float) -> None:
            pair = (exchange, symbol)
            if pair in seen:
                return
            seen.add(pair)
            hits.append(
                SymbolHit(
                    symbol=symbol,
                    exchange=exchange,
                    company_name=cname,
                    score=score,
                )
            )

        for symbol, exchange, cname in self._by_norm.get(key, []):
            add(symbol, exchange, cname, 1.0)

        if len(key) >= 4:
            for cand, rows in self._by_norm.items():
                if cand == key:
                    continue
                if key in cand or cand in key:
                    shorter, longer = (
                        (key, cand) if len(key) <= len(cand) else (cand, key)
                    )
                    if len(shorter) < 5 and not longer.startswith(shorter):
                        continue
                    score = len(shorter) / max(len(longer), 1)
                    if score < 0.55:
                        continue
                    for symbol, exchange, cname in rows:
                        add(symbol, exchange, cname, round(score, 3))

        hits.sort(key=lambda h: (-h.score, 0 if h.exchange == "NSE" else 1, h.symbol))
        return hits[:max_hits]

    def resolve_names(self, names: list[str]) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        for name in names:
            for hit in self.resolve_name(name, max_hits=1):
                if hit.symbol not in seen:
                    seen.add(hit.symbol)
                    symbols.append(hit.symbol)
        return symbols

    def has_symbol(self, symbol: str) -> bool:
        return str(symbol or "").strip().upper() in self._by_symbol

    def candidates_for_article(
        self,
        *,
        headline: str,
        content: str,
        source_companies: list[str],
        limit: int = 50,
    ) -> list[SymbolHit]:
        found: dict[tuple[str, str], SymbolHit] = {}

        def add_hits(hits: list[SymbolHit]) -> None:
            for hit in hits:
                key = (hit.exchange, hit.symbol)
                prev = found.get(key)
                if prev is None or hit.score > prev.score:
                    found[key] = hit

        for name in source_companies:
            add_hits(self.resolve_name(name, max_hits=3))

        add_hits(self.resolve_name(headline, max_hits=5))

        blob = normalize_name(f"{headline} {content[:2500]}")
        tokens = [t for t in blob.split() if len(t) >= 4]
        token_set = set(tokens)
        for key, rows in self._by_norm.items():
            key_tokens = key.split()
            if not key_tokens:
                continue
            overlap = sum(1 for t in key_tokens if t in token_set)
            if overlap == 0:
                continue
            if len(key_tokens) >= 2:
                strong = sum(
                    1 for t in key_tokens if len(t) >= 5 and t in token_set
                )
                if strong < 1 or overlap < max(2, len(key_tokens) // 2):
                    continue
            elif key_tokens[0] not in token_set or len(key_tokens[0]) < 5:
                continue
            score = overlap / max(len(key_tokens), 1)
            for symbol, exchange, cname in rows:
                add_hits(
                    [
                        SymbolHit(
                            symbol=symbol,
                            exchange=exchange,
                            company_name=cname,
                            score=round(min(0.99, score), 3),
                        )
                    ]
                )

        ranked = sorted(
            found.values(),
            key=lambda h: (-h.score, 0 if h.exchange == "NSE" else 1, h.symbol),
        )
        out: list[SymbolHit] = []
        seen_sym: set[str] = set()
        for hit in ranked:
            if hit.symbol in seen_sym:
                continue
            seen_sym.add(hit.symbol)
            out.append(hit)
            if len(out) >= limit:
                break
        return out


@lru_cache(maxsize=1)
def get_resolver() -> SymbolResolver:
    return SymbolResolver.from_iceberg()


def clear_resolver_cache() -> None:
    get_resolver.cache_clear()
