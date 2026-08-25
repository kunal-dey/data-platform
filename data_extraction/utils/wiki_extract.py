import time

import requests

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "DataPlatformBot/1.0 (https://github.com/wealth-management-system-v2)"
MIN_REQUEST_INTERVAL_SECONDS = 1.0
MAX_RETRIES = 5

_last_request_at = 0.0


def _wait_for_rate_limit() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)


def get_wikipedia_text(title: str) -> str:
    """Fetch full Wikipedia article text, with throttling and 429 retries."""
    global _last_request_at

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,
    }
    headers = {"User-Agent": WIKI_USER_AGENT}

    for attempt in range(MAX_RETRIES):
        _wait_for_rate_limit()

        try:
            response = requests.get(
                WIKI_API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            _last_request_at = time.monotonic()

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", min(60, 2**attempt)))
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            page = next(iter(response.json()["query"]["pages"].values()))
            return page.get("extract", "")
        except requests.HTTPError as exc:
            if (
                exc.response is not None
                and exc.response.status_code == 429
                and attempt < MAX_RETRIES - 1
            ):
                time.sleep(min(60, 2**attempt))
                continue
            return ""
        except requests.RequestException:
            return ""

    return ""
