#!/usr/bin/env python3
"""Download KuniJiban soil-test XML files through the public viewer endpoints.

The KuniJiban viewer currently exposes a JSON search endpoint and an XML
reference endpoint. This script uses those endpoints sequentially and keeps a
CSV manifest so interrupted downloads can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.kunijiban.pwri.go.jp/viewer"
SEARCH_URL = f"{BASE_URL}/server/search.php"
XML_URL_TEMPLATE = f"{BASE_URL}/refer/?data=soiltest&type=xml&id={{}}"
DEFAULT_BBOX = "37.60,136.50,36.90,137.50"  # north, west, south, east; Noto area
DEFAULT_OUTPUT_DIR = Path("input_xml/strength")
DEFAULT_METADATA_DIR = Path("metadata")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_DELAY = 2.5
PAGE_SIZE = 30
USER_AGENT = "KuniJibanResearchDownloader/0.1 (sequential, low-rate access)"
CSV_COLUMNS = [
    "boring_id", "code", "boring_name", "latitude", "longitude", "surface_z",
    "boring_length", "project_name", "survey_name", "client_name",
    "has_soil_test", "has_strength_xml", "xml_url", "download_status", "local_file",
]

LOGGER = logging.getLogger("kunijiban.download")


def configure_logging(log_dir: Path) -> None:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)
    file_handler = logging.FileHandler(log_dir / "download.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    error_handler = logging.FileHandler(log_dir / "error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    LOGGER.addHandler(error_handler)


def create_session() -> requests.Session:
    """Create an HTTP session with conservative browser-like headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, application/xml, text/xml;q=0.9, */*;q=0.8",
        "Referer": f"{BASE_URL}/",
    })
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None,
    timeout: float,
    retries: int,
) -> requests.Response:
    """Perform a request with exponential backoff for transient failures."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"transient HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait_seconds = min(60.0, 2.0 ** attempt)
            LOGGER.warning("Request failed (%s); retrying in %.1fs: %s", type(exc).__name__, wait_seconds, url)
            time.sleep(wait_seconds)
    raise RuntimeError(f"request failed after {retries + 1} attempts: {url}: {last_error}")


def search_records(
    session: requests.Session,
    bbox: str,
    timeout: float,
    retries: int,
    delay: float,
    max_records: int | None,
) -> list[dict[str, Any]]:
    """Search all soil-test records in a bbox, following API pagination."""
    records: list[dict[str, Any]] = []
    page = 1
    total_pages: int | None = None
    while total_pages is None or page <= total_pages:
        params = {"soiltest_result": "1", "bbox": bbox, "page": str(page)}
        response = request_with_retry(session, SEARCH_URL, params, timeout, retries)
        payload = response.json()
        header = payload.get("header", {})
        if not header.get("success"):
            raise RuntimeError(f"KuniJiban search failed: {header.get('message', '')}")
        data = payload.get("data", {})
        meta = data.get("meta", {})
        total_pages = int(meta.get("pages", 0))
        values = data.get("values", [])
        LOGGER.info("Search page %d/%d: %d records", page, total_pages, len(values))
        records.extend(value for value in values if isinstance(value, dict))
        if max_records is not None and len(records) >= max_records:
            return records[:max_records]
        if page >= total_pages or not values:
            break
        page += 1
        time.sleep(delay)
    return records


def safe_filename(record: dict[str, Any], content_disposition: str | None = None) -> str:
    """Choose an ASCII-safe deterministic filename for a downloaded XML."""
    match = re.search(r"filename=\"?([^\";]+)", content_disposition or "", re.IGNORECASE)
    original = match.group(1).strip() if match else ""
    suffix = Path(original).suffix.lower()
    if suffix != ".xml":
        suffix = ".xml"
    code = str(record.get("code") or record.get("id") or "unknown")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", code).strip("._")
    return f"{cleaned}_{record.get('id')}{suffix}"


def validate_xml(content: bytes) -> None:
    """Validate a KuniJiban XML payload, including its Shift_JIS encoding."""
    if not content.strip():
        raise ValueError("empty XML response")
    decoded = content.decode("shift_jis")
    ET.fromstring(decoded)


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load previous manifest rows indexed by boring ID."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["boring_id"]: row for row in csv.DictReader(stream) if row.get("boring_id")}


def write_manifest(path: Path, rows: dict[str, dict[str, str]]) -> None:
    """Write the resumable metadata manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows, key=lambda value: int(value) if value.isdigit() else value))


def record_to_manifest(record: dict[str, Any], xml_url: str) -> dict[str, str]:
    """Convert an API record to the stable CSV metadata schema."""
    return {
        "boring_id": str(record.get("id", "")),
        "code": str(record.get("code", "")),
        "boring_name": str(record.get("boring_name", "")),
        "latitude": str(record.get("latitude", "")),
        "longitude": str(record.get("longitude", "")),
        "surface_z": str(record.get("boring_elevation", "")),
        "boring_length": str(record.get("boring_length", "")),
        "project_name": str(record.get("project_name", "")),
        "survey_name": str(record.get("survey_name", "")),
        "client_name": str(record.get("client_name", "")),
        "has_soil_test": "1",
        "has_strength_xml": "1" if int(record.get("soiltest_xml_url") or 0) > 0 else "0",
        "xml_url": xml_url,
        "download_status": "pending",
        "local_file": "",
    }


def download_records(
    records: list[dict[str, Any]],
    session: requests.Session,
    output_dir: Path,
    manifest_path: Path,
    timeout: float,
    retries: int,
    delay: float,
    manifest: dict[str, dict[str, str]],
) -> None:
    """Download XML records sequentially and persist progress after each item."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        boring_id = str(record.get("id", ""))
        xml_url = XML_URL_TEMPLATE.format(boring_id)
        row = manifest.get(boring_id, record_to_manifest(record, xml_url))
        destination = output_dir / safe_filename(record)
        row["xml_url"] = xml_url
        if row.get("download_status") == "success" and destination.is_file() and destination.stat().st_size > 0:
            row["download_status"] = "skipped"
            row["local_file"] = str(destination)
            manifest[boring_id] = row
            write_manifest(manifest_path, manifest)
            LOGGER.info("[%d/%d] skipped existing XML for id=%s", index, len(records), boring_id)
            continue
        if int(record.get("soiltest_xml_url") or 0) <= 0:
            row["download_status"] = "no_xml"
            manifest[boring_id] = row
            write_manifest(manifest_path, manifest)
            continue
        try:
            response = request_with_retry(session, xml_url, None, timeout, retries)
            validate_xml(response.content)
            destination.write_bytes(response.content)
            row["download_status"] = "success"
            row["local_file"] = str(destination)
            LOGGER.info("[%d/%d] downloaded id=%s -> %s", index, len(records), boring_id, destination)
        except (RuntimeError, ValueError, UnicodeError, ET.ParseError, OSError) as exc:
            row["download_status"] = "failed"
            row["local_file"] = ""
            LOGGER.error("[%d/%d] failed id=%s url=%s: %s", index, len(records), boring_id, xml_url, exc)
        manifest[boring_id] = row
        write_manifest(manifest_path, manifest)
        if index < len(records):
            time.sleep(delay)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", default=DEFAULT_BBOX, help="north,west,south,east in decimal degrees")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="seconds between requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=None, help="limit records for a test run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Search and download soil-test XML files."""
    args = parse_args(argv)
    if args.delay < 0 or args.max_records == 0:
        raise ValueError("delay must be non-negative and max-records must be positive")
    configure_logging(args.log_dir)
    manifest_path = args.metadata_dir / "boring_list.csv"
    manifest = load_manifest(manifest_path)
    session = create_session()
    LOGGER.info("Searching soil-test records in bbox=%s", args.bbox)
    records = search_records(session, args.bbox, args.timeout, args.retries, args.delay, args.max_records)
    LOGGER.info("Found %d soil-test records", len(records))
    download_records(records, session, args.output_dir, manifest_path, args.timeout, args.retries, args.delay, manifest)
    LOGGER.info("Finished. Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
        LOGGER.error("Stopped: %s", exc)
        sys.exit(1)
