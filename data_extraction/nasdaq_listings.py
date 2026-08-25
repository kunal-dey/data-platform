import csv
import io
import os
import sys
from pathlib import Path

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))

from utils.wiki_extract import get_wikipedia_text
import dlt
import requests

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
# Load variables from .env
load_dotenv(_DATA_EXTRACTION_DIR.parent / ".env")

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
DEFAULT_DATASET = "nasdaq_listed"
WIKI_S3_PREFIX = "company_wiki"


def _s3_bucket_name() -> str:
    return os.getenv("S3_LAKE_BASE", "s3://data-platform-427899482524-ap-south-1-an").removeprefix(
        "s3://"
    ).rstrip("/")


s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

@dlt.resource(
    name="nasdaq_listed",
    write_disposition="replace",
    table_format="iceberg",
)
def nasdaq_listed():
    response = requests.get(NASDAQ_LISTED_URL, timeout=60)
    response.raise_for_status()

    reader = csv.DictReader(io.StringIO(response.text), delimiter="|")

    for row in reader:
        yield row


def extract_company_wiki(item):
    symbol = item.get("Symbol", "").strip()
    security_name = item.get("Security Name", "").strip()
    wiki_s3_uri = ""

    # if symbol and security_name and not security_name.startswith("File Creation"):
    #     bucket_name = _s3_bucket_name()
    #     s3_key = f"{WIKI_S3_PREFIX}/{symbol}.txt"
    #     wiki_s3_uri = f"s3://{bucket_name}/{s3_key}"

    #     try:
    #         s3.head_object(Bucket=bucket_name, Key=s3_key)
    #     except ClientError:
    #         page_title = security_name.split(" - ")[0].strip()
    #         text = get_wikipedia_text(page_title).strip()
    #         if text:
    #             s3.put_object(
    #                 Bucket=bucket_name,
    #                 Key=s3_key,
    #                 Body=text.encode("utf-8"),
    #                 ContentType="text/plain; charset=utf-8",
    #             )
    #         else:
    #             wiki_s3_uri = ""
    return {**item, "wiki_s3_uri": wiki_s3_uri}

@dlt.source
def nasdaq_listings_source():
    return nasdaq_listed().add_map(extract_company_wiki)

if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)

    pipeline = dlt.pipeline(
        pipeline_name="nasdaq_listed",
        destination="filesystem",
        dataset_name=DEFAULT_DATASET,
    )
    load_info = pipeline.run(nasdaq_listed())
    print(load_info)
