"""Fetch NSE/BSE equity masters via nselib + httpx."""

from __future__ import annotations

import os

import httpx
import pandas as pd
from nselib.capital_market import equity_list
from tenacity import retry, stop_after_attempt, wait_fixed

BSE_EQUITY_API = os.getenv(
    "BSE_EQUITY_API",
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    "?Group=&Scripcode=&industry=&segment=Equity&status=Active",
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json,*/*",
}
COLS = [
    "exchange",
    "symbol",
    "company_name",
    "isin",
    "series",
    "listing_date",
    "face_value",
    "exchange_code",
]


def fetch_nse() -> pd.DataFrame:
    df = equity_list()
    df.columns = df.columns.str.strip()
    return (
        df.rename(
            columns={
                "SYMBOL": "symbol",
                "NAME OF COMPANY": "company_name",
                "SERIES": "series",
                "DATE OF LISTING": "listing_date",
                "FACE VALUE": "face_value",
            }
        )
        .assign(exchange="NSE", isin="", exchange_code="")
        .loc[:, COLS]
        .drop_duplicates(subset=["exchange", "symbol"])
        .reset_index(drop=True)
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def fetch_bse() -> pd.DataFrame:
    r = httpx.get(BSE_EQUITY_API, headers=HEADERS, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    name = df["Issuer_Name"].where(df["Issuer_Name"].astype(str).str.strip().ne(""), df["Scrip_Name"])
    out = pd.DataFrame(
        {
            "exchange": "BSE",
            "symbol": df["scrip_id"].astype(str).str.strip().str.upper(),
            "company_name": name.map(lambda x: " ".join(str(x or "").split())),
            "isin": df["ISIN_NUMBER"].astype(str).str.strip().str.upper(),
            "series": df["GROUP"].astype(str).str.strip(),
            "listing_date": "",
            "face_value": pd.to_numeric(df["FACE_VALUE"], errors="coerce"),
            "exchange_code": df["SCRIP_CD"].astype(str).str.strip(),
        }
    )
    out = out[out["symbol"].ne("") & out["company_name"].ne("")]
    out = out[out["isin"].str.startswith("INE", na=False)]
    return out.drop_duplicates(subset=["exchange", "symbol"]).reset_index(drop=True)


def fetch_universe() -> pd.DataFrame:
    return (
        pd.concat([fetch_nse(), fetch_bse()], ignore_index=True)
        .drop_duplicates(subset=["exchange", "symbol"])
        .reset_index(drop=True)
    )
