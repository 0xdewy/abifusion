#!/usr/bin/env python3
"""Download Sourcify v2 parquet exports.

Sourcify publishes public parquet exports at:

    https://export.sourcify.dev/v2/<table>/<file>.parquet

This script uses the export bucket listing API, downloads selected tables, and
stores them as:

    data/sourcify/<table>/<file>.parquet
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import quote

import requests


EXPORT_BASE_URL = "https://export.sourcify.dev"
DEFAULT_TABLES = [
    "contracts",
    "contract_deployments",
    "compiled_contracts",
    "code",
]
ALL_TABLES = [
    "code",
    "contracts",
    "contract_deployments",
    "compiled_contracts",
    "compiled_contracts_sources",
    "sources",
    "verified_contracts",
    "sourcify_matches",
    "signatures",
    "compiled_contracts_signatures",
]


def parse_table_list(value: str) -> List[str]:
    if value == "default":
        return DEFAULT_TABLES
    if value == "all":
        return ALL_TABLES
    tables = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(tables) - set(ALL_TABLES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown Sourcify table(s): {', '.join(unknown)}"
        )
    return tables


def _xml_text(element: ET.Element, tag_name: str) -> str:
    return next(
        (
            child.text or ""
            for child in element
            if child.tag.rsplit("}", 1)[-1] == tag_name
        ),
        "",
    )


def list_export_keys(table: str, session: requests.Session) -> List[str]:
    """List parquet object keys for one Sourcify v2 table."""
    prefix = f"v2/{table}/"
    response = session.get(
        EXPORT_BASE_URL,
        params={"prefix": prefix},
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    keys = []
    for contents in root.iter():
        if contents.tag.rsplit("}", 1)[-1] != "Contents":
            continue
        key = _xml_text(contents, "Key")
        if key.endswith(".parquet"):
            keys.append(key)
    return sorted(keys)


def download_key(
    key: str,
    output_dir: Path,
    session: requests.Session,
    overwrite: bool = False,
) -> Path:
    """Download one export object key under output_dir."""
    table = key.split("/")[1]
    filename = Path(key).name
    target_dir = output_dir / table
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    if target.exists() and not overwrite:
        print(f"skip existing: {target}")
        return target

    url = f"{EXPORT_BASE_URL}/{quote(key)}"
    tmp_target = target.with_suffix(target.suffix + ".part")
    print(f"download: {url} -> {target}")

    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp_target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_target.replace(target)
    return target


def iter_keys(
    tables: Iterable[str],
    session: requests.Session,
    max_files_per_table: Optional[int] = None,
) -> Iterable[str]:
    for table in tables:
        keys = list_export_keys(table, session)
        if max_files_per_table is not None:
            keys = keys[:max_files_per_table]
        print(f"{table}: {len(keys)} parquet file(s)")
        yield from keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        type=parse_table_list,
        default=DEFAULT_TABLES,
        help=(
            "Comma-separated table list, 'default', or 'all'. "
            f"Default tables: {','.join(DEFAULT_TABLES)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/sourcify",
        help="Directory where table folders will be written",
    )
    parser.add_argument(
        "--max-files-per-table",
        type=int,
        help="Download only the first N parquet files per table",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List matching export files without downloading",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    session = requests.Session()
    session.headers.update({"User-Agent": "abi-reconstructor/0.1"})

    try:
        keys = list(iter_keys(args.tables, session, args.max_files_per_table))
        if args.list_only:
            for key in keys:
                print(key)
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        for key in keys:
            download_key(key, output_dir, session, overwrite=args.overwrite)
        return 0
    except requests.RequestException as e:
        print(f"Sourcify export request failed: {e}", file=sys.stderr)
        return 1
    except ET.ParseError as e:
        print(f"Could not parse Sourcify export listing XML: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
