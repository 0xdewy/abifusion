"""Local SQLite 4byte database for selector→signature mappings."""

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


@dataclass
class Signature:
    """Represents a function signature."""

    selector: str
    text_signature: str
    source: str = "4byte.directory"
    created_at: Optional[str] = None
    id: Optional[int] = None


class FourByteDatabase:
    """Local SQLite database for 4byte selector→signature mappings."""

    FOURBYTE_API = "https://www.4byte.directory/api/v1/signatures/"
    FOURBYTE_EVENTS_API = "https://www.4byte.directory/api/v1/event-signatures/"
    STANDARD_SIGNATURES: Dict[str, List[str]] = {
        "a9059cbb": ["transfer(address,uint256)"],
        "095ea7b3": ["approve(address,uint256)"],
        "70a08231": ["balanceOf(address)"],
        "23b872dd": ["transferFrom(address,address,uint256)"],
        "18160ddd": ["totalSupply()"],
        "06fdde03": ["name()"],
        "95d89b41": ["symbol()"],
        "313ce567": ["decimals()"],
        "dd62ed3e": ["allowance(address,address)"],
        "6352211e": ["ownerOf(uint256)"],
        "42842e0e": ["safeTransferFrom(address,address,uint256)"],
        "b88d4fde": ["safeTransferFrom(address,address,uint256,bytes)"],
        "081812fc": ["getApproved(uint256)"],
        "a22cb465": ["setApprovalForAll(address,bool)"],
        "e985e9c5": ["isApprovedForAll(address,address)"],
        "40c10f19": ["mint(address,uint256)"],
        "d0e30db0": ["deposit()"],
        "2e1a7d4d": ["withdraw(uint256)"],
        "8da5cb5b": ["owner()"],
        "f2fde38b": ["transferOwnership(address)"],
        "715018a6": ["renounceOwnership()"],
        "8456cb59": ["pause()"],
        "3f4ba83a": ["unpause()"],
        "5c975abb": ["paused()"],
        "36568abe": ["hasRole(bytes32,address)"],
        "2f2ff15d": ["grantRole(bytes32,address)"],
        "d547741f": ["revokeRole(bytes32,address)"],
        "91d14854": ["renounceRole(bytes32,address)"],
        "248a9ca3": ["getRoleAdmin(bytes32)"],
        "a217fddf": ["DEFAULT_ADMIN_ROLE()"],
        "f242432a": ["safeTransferFrom(address,address,uint256,uint256,bytes)"],
        "2eb2c2d6": ["safeBatchTransferFrom(address,address,uint256[],uint256[],bytes)"],
        "00fdd58e": ["balanceOf(address,uint256)"],
        "4e1273f4": ["balanceOfBatch(address[],uint256[])"],
        "3b6c8fc1": ["setURI(string,bool)"],
        "e8e33700": ["setRoyaltyReceiver(address,uint256)"],
        "c7a5f67a": ["getRoyaltyReceiver(uint256)"],
        "efa3d2b0": ["getCreator(uint256)"],
        "f4e9c280": ["mint(address,uint256,string)"],
        "52d6c89d": ["burn(uint256,uint256)"],
        "b830e2af": ["tokenURI(uint256)"],
        "06b7f4bd": ["initialize(address)"],
        "4e2709e3": ["receive()"],
    }

    def __init__(self, db_path: str = "./cache/4byte.db", cache_dir: str = "./cache/signatures"):
        """Initialize the 4byte database.

        Args:
            db_path: Path to SQLite database file
            cache_dir: Directory for API response caching
        """
        self.db_path = db_path
        self.cache_dir = cache_dir
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "ABI-Reconstructor/1.0",
            "Accept": "application/json",
        })
        self._last_request_time = 0
        self._min_request_interval = 0.5

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._initialize_schema()
        return self._conn

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                selector TEXT NOT NULL,
                text_signature TEXT NOT NULL,
                source TEXT DEFAULT '4byte.directory',
                created_at TEXT,
                UNIQUE(selector, text_signature)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_selector ON signatures(selector)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()

        self._populate_standard_signatures()

    def _populate_standard_signatures(self) -> None:
        """Populate database with standard signatures."""
        for selector, sigs in self.STANDARD_SIGNATURES.items():
            for sig in sigs:
                try:
                    self._conn.execute("""
                        INSERT OR IGNORE INTO signatures (selector, text_signature, source, created_at)
                        VALUES (?, ?, 'standard', ?)
                    """, (selector, sig, datetime.now().isoformat()))
                except sqlite3.Error as e:
                    logging.debug(f"Database error inserting standard signature: {e}")

    def _rate_limit(self) -> None:
        """Apply rate limiting between API requests."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _cache_get(self, selector: str) -> Optional[List[Dict]]:
        """Get cached API response."""
        cache_key = hashlib.md5(selector.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < 2592000:
                    return data.get("signatures", [])
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _cache_set(self, selector: str, signatures: List[Dict]) -> None:
        """Cache API response."""
        cache_key = hashlib.md5(selector.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "selector": selector,
                    "signatures": signatures,
                }, f, indent=2)
        except IOError:
            pass

    def fetch_from_4byte_api(self, selector: str) -> List[Dict]:
        """Fetch signatures from 4byte.directory API.

        Args:
            selector: Function selector (hex string without 0x)

        Returns:
            List of signature dictionaries
        """
        cached = self._cache_get(selector)
        if cached is not None:
            return cached

        self._rate_limit()

        try:
            response = self._session.get(
                self.FOURBYTE_API,
                params={"hex_signature": f"0x{selector}"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            signatures = []
            for item in data.get("results", []):
                signatures.append({
                    "text_signature": item.get("text_signature", ""),
                    "hex_signature": item.get("hex_signature", ""),
                    "id": item.get("id"),
                    "created_at": item.get("created_at", ""),
                    "source": "4byte.directory",
                })

            self._cache_set(selector, signatures)
            return signatures

        except requests.exceptions.RequestException as e:
            print(f"Warning: API request failed for 0x{selector}: {e}")
            return []

    def fetch_event_signature_from_api(self, topic0: str) -> List[Dict]:
        """Fetch event signature for a topic from 4byte.directory events API.

        Args:
            topic0: Event topic0 (32-byte keccak256 hash as hex string)

        Returns:
            List of event signature dictionaries
        """
        cache_key = hashlib.md5(topic0.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"event_{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < 2592000:
                    return data.get("signatures", [])
            except (json.JSONDecodeError, IOError):
                pass

        self._rate_limit()

        try:
            response = self._session.get(
                self.FOURBYTE_EVENTS_API,
                params={"hex_topic0": f"0x{topic0}"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            signatures = []
            for item in data.get("results", []):
                signatures.append({
                    "text_signature": item.get("text_signature", ""),
                    "hex_topic0": item.get("hex_topic0", ""),
                    "id": item.get("id"),
                    "created_at": item.get("created_at", ""),
                    "source": "4byte.directory",
                })

            with open(cache_path, "w") as f:
                json.dump({"timestamp": time.time(), "topic0": topic0, "signatures": signatures}, f, indent=2)
            return signatures

        except requests.exceptions.RequestException as e:
            print(f"Warning: Event API request failed for 0x{topic0}: {e}")
            return []

    def insert_signature(self, selector: str, text_signature: str, source: str = "4byte.directory") -> bool:
        """Insert a signature into the database.

        Args:
            selector: Function selector (8 hex chars)
            text_signature: Function signature string

        Returns:
            True if inserted, False otherwise
        """
        try:
            conn = self._get_connection()
            conn.execute("""
                INSERT OR IGNORE INTO signatures (selector, text_signature, source, created_at)
                VALUES (?, ?, ?, ?)
            """, (selector, text_signature, source, datetime.now().isoformat()))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logging.debug(f"Database error inserting signature: {e}")
            return False

    def get_signatures(self, selector: str) -> List[Signature]:
        """Get all signatures for a selector from local database.

        Args:
            selector: Function selector (8 hex chars)

        Returns:
            List of Signature objects
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT id, selector, text_signature, source, created_at
            FROM signatures
            WHERE selector = ?
            ORDER BY source = 'standard' DESC, created_at DESC
        """, (selector,))

        return [
            Signature(
                selector=row["selector"],
                text_signature=row["text_signature"],
                source=row["source"],
                created_at=row["created_at"],
                id=row["id"],
            )
            for row in cursor
        ]

    def lookup(self, selector: str, fetch_api: bool = True) -> List[Signature]:
        """Look up signatures for a selector, with optional API fallback.

        Args:
            selector: Function selector (8 hex chars)
            fetch_api: Whether to fetch from API if not in local DB

        Returns:
            List of Signature objects
        """
        signatures = self.get_signatures(selector)

        if not signatures and fetch_api:
            api_sigs = self.fetch_from_4byte_api(selector)
            for sig_dict in api_sigs:
                text_sig = sig_dict.get("text_signature", "")
                if text_sig:
                    self.insert_signature(selector, text_sig, "4byte.directory")
                    signatures.append(Signature(
                        selector=selector,
                        text_signature=text_sig,
                        source="4byte.directory",
                        created_at=sig_dict.get("created_at"),
                    ))

        return signatures

    def get_best_signature(self, selector: str) -> Optional[str]:
        """Get the best (most likely correct) signature for a selector.

        Args:
            selector: Function selector

        Returns:
            Best signature string or None
        """
        if selector in self.STANDARD_SIGNATURES:
            return self.STANDARD_SIGNATURES[selector][0]

        signatures = self.lookup(selector, fetch_api=True)
        if signatures:
            for sig in signatures:
                if sig.source == "standard":
                    return sig.text_signature
            return signatures[0].text_signature
        return None

    def populate_from_api(self, selectors: Optional[List[str]] = None, batch_size: int = 100) -> Dict[str, int]:
        """Populate database by fetching from 4byte.directory API.

        Args:
            selectors: List of selectors to fetch (None = paginate through API)
            batch_size: Number of selectors per batch

        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0, "fetched": 0}

        if selectors is None:
            page = 1
            while True:
                self._rate_limit()
                try:
                    response = self._session.get(
                        self.FOURBYTE_API,
                        params={"page": page},
                        timeout=10
                    )
                    response.raise_for_status()
                    data = response.json()
                    api_results = data.get("results", [])
                    if not api_results:
                        break

                    for item in api_results:
                        hex_sig = item.get("hex_signature", "")
                        if hex_sig.startswith("0x"):
                            selector = hex_sig[2:10]
                        else:
                            selector = hex_sig[:8]
                        text_sig = item.get("text_signature", "")
                        if selector and text_sig:
                            self.insert_signature(selector, text_sig, "4byte.directory")
                            results["fetched"] += 1

                    if not data.get("next"):
                        break
                    page += 1

                    if page > 1000:
                        break

                except requests.exceptions.RequestException as e:
                    print(f"Warning: API pagination failed at page {page}: {e}")
                    break

            results["success"] = results["fetched"]
            return results

        for selector in selectors:
            existing = self.get_signatures(selector)
            if existing and any(s.source == "4byte.directory" for s in existing):
                results["success"] += 1
                continue

            sigs = self.fetch_from_4byte_api(selector)
            if sigs:
                for sig_dict in sigs:
                    text_sig = sig_dict.get("text_signature", "")
                    if text_sig:
                        self.insert_signature(selector, text_sig, "4byte.directory")
                results["success"] += 1
            else:
                results["failed"] += 1

            time.sleep(0.5)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as total, COUNT(DISTINCT selector) as unique_selectors FROM signatures")
        row = cursor.fetchone()

        cursor = conn.execute("SELECT COUNT(*) as count FROM signatures WHERE source = 'standard'")
        standard_count = cursor.fetchone()["count"]

        return {
            "total_signatures": row["total"],
            "unique_selectors": row["unique_selectors"],
            "standard_signatures": standard_count,
            "db_path": self.db_path,
        }

    def export_json(self) -> Dict[str, List[str]]:
        """Export database as JSON dictionary."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT selector, text_signature FROM signatures ORDER BY selector")
        result: Dict[str, List[str]] = {}
        for row in cursor:
            sel = row["selector"]
            if sel not in result:
                result[sel] = []
            result[sel].append(row["text_signature"])
        return result

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Cleanup on destruction."""
        self.close()