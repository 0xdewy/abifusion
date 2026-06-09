"""Signature lookup from 4byte.directory for ABI reconstructor."""

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class SignatureLookup:
    """Look up function signatures from 4byte.directory."""

    # 4byte.directory API endpoint
    FOURBYTE_API = "https://www.4byte.directory/api/v1/signatures/"

    # Common standard signatures (for prioritization)
    STANDARD_SIGNATURES = {
        # ERC20
        "a9059cbb": ["transfer(address,uint256)"],
        "095ea7b3": ["approve(address,uint256)"],
        "70a08231": ["balanceOf(address)"],
        "23b872dd": ["transferFrom(address,address,uint256)"],
        "18160ddd": ["totalSupply()"],
        "06fdde03": ["name()"],
        "95d89b41": ["symbol()"],
        "313ce567": ["decimals()"],
        "dd62ed3e": ["allowance(address,address)"],
        # ERC721
        "6352211e": ["ownerOf(uint256)"],
        "42842e0e": ["safeTransferFrom(address,address,uint256)"],
        "b88d4fde": ["safeTransferFrom(address,address,uint256,bytes)"],
        # Note: "23b872dd" and "095ea7b3" already defined in ERC20 section
        "081812fc": ["getApproved(uint256)"],
        "a22cb465": ["setApprovalForAll(address,bool)"],
        "e985e9c5": ["isApprovedForAll(address,address)"],
        # Common utilities
        "cdffacc6": ["getFunctionImplementation(bytes4)"],
        "52ef6b2c": ["addup(uint256,uint256)"],
        "019c8af1": ["mint()"],
        "40c10f19": ["mint(address,uint256)"],
        "d0e30db0": ["deposit()"],
        "2e1a7d4d": ["withdraw(uint256)"],
    }

    def __init__(self, cache_dir: str = "./cache/signatures"):
        """Initialize signature lookup.

        Args:
            cache_dir: Directory to cache signature lookups
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Request session with headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "ABI-Reconstructor/1.0",
                "Accept": "application/json",
            }
        )

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

    def _get_cache_key(self, selector: str) -> str:
        """Generate cache key for selector."""
        return f"sig_{selector}"

    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cached signature."""
        filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
        return os.path.join(self.cache_dir, filename)

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Load signatures from cache."""
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)

                # Check if cache is still valid (30 days for signatures)
                if time.time() - data.get("timestamp", 0) < 2592000:
                    return data.get("signatures", [])
            except Exception:
                pass

        return None

    def _save_to_cache(self, cache_key: str, signatures: List[Dict]):
        """Save signatures to cache."""
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            "timestamp": time.time(),
            "cache_key": cache_key,
            "signatures": signatures,
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass  # Cache write failure is not critical

    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time_to_wait = self.min_request_interval - time_since_last
            time.sleep(time_to_wait)

        self.last_request_time = time.time()

    def lookup_4byte(self, selector: str) -> List[Dict]:
        """Look up signatures from 4byte.directory.

        Args:
            selector: Function selector (hex string without 0x)

        Returns:
            List of signature dictionaries with metadata
        """
        cache_key = self._get_cache_key(selector)

        # Check cache first
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Apply rate limiting
        self._rate_limit()

        # Query 4byte.directory
        params = {"hex_signature": f"0x{selector}"}

        try:
            response = self.session.get(self.FOURBYTE_API, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            signatures = []
            for item in data.get("results", []):
                signature = {
                    "text_signature": item.get("text_signature", ""),
                    "hex_signature": item.get("hex_signature", ""),
                    "id": item.get("id"),
                    "created_at": item.get("created_at", ""),
                    "source": "4byte.directory",
                }
                signatures.append(signature)

            # Save to cache
            self._save_to_cache(cache_key, signatures)

            return signatures

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to query 4byte.directory for 0x{selector}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from 4byte.directory for 0x{selector}: {e}")
            return []

    OPENCHAIN_API = "https://api.openchain.xyz/signature-database/v1/lookup"

    def lookup_openchain(self, selector: str) -> List[Dict]:
        """Look up signatures from openchain.xyz (samczsun's signature DB).

        Higher coverage and far less collision spam than 4byte.directory, which
        makes it the preferred candidate source for fusion. Cached separately.

        Args:
            selector: Function selector (hex string without 0x).

        Returns:
            List of signature dicts with ``text_signature``.
        """
        cache_key = f"oc_{selector}"
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()
        params = {"function": f"0x{selector}", "filter": "true"}
        try:
            response = self.session.get(self.OPENCHAIN_API, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = (
                data.get("result", {}).get("function", {}).get(f"0x{selector}") or []
            )
            signatures = [
                {
                    "text_signature": item.get("name", ""),
                    "hex_signature": f"0x{selector}",
                    "source": "openchain.xyz",
                }
                for item in results
                if item.get("name")
            ]
            self._save_to_cache(cache_key, signatures)
            return signatures
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to query openchain for 0x{selector}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from openchain for 0x{selector}: {e}")
            return []

    def get_standard_signatures(self, selector: str) -> List[str]:
        """Get standard signatures for selector from built-in database.

        Args:
            selector: Function selector (hex string without 0x)

        Returns:
            List of standard signatures
        """
        return self.STANDARD_SIGNATURES.get(selector, [])

    def lookup_signatures(self, selector: str) -> List[Dict]:
        """Look up signatures from all sources.

        Args:
            selector: Function selector (hex string without 0x)

        Returns:
            List of signature dictionaries from all sources
        """
        all_signatures = []

        # Add standard signatures first (highest priority)
        standard_sigs = self.get_standard_signatures(selector)
        for sig in standard_sigs:
            all_signatures.append(
                {
                    "text_signature": sig,
                    "hex_signature": f"0x{selector}",
                    "source": "standard",
                    "priority": 1,  # Highest priority
                }
            )

        # Query 4byte.directory
        fourbyte_sigs = self.lookup_4byte(selector)
        for sig in fourbyte_sigs:
            sig["priority"] = 2  # Medium priority
            all_signatures.append(sig)

        # Remove duplicates (same text_signature)
        unique_signatures = []
        seen = set()

        for sig in all_signatures:
            text_sig = sig.get("text_signature", "")
            if text_sig and text_sig not in seen:
                seen.add(text_sig)
                unique_signatures.append(sig)

        # Sort by priority (standard first), then by creation date (newest first)
        unique_signatures.sort(
            key=lambda x: (
                x.get("priority", 3),
                x.get("created_at", ""),  # Newer first (empty string sorts first)
            )
        )

        return unique_signatures

    def get_best_signature(self, selector: str) -> Optional[Tuple[str, float]]:
        """Get the best signature for a selector with confidence score.

        Args:
            selector: Function selector (hex string without 0x)

        Returns:
            Tuple of (signature, confidence) or None if no signatures found
        """
        signatures = self.lookup_signatures(selector)

        if not signatures:
            return None

        # Get the first (highest priority) signature
        best_sig = signatures[0]
        text_sig = best_sig.get("text_signature", "")

        if not text_sig:
            return None

        # Calculate confidence score (0.0 to 1.0)
        confidence = self._calculate_confidence(signatures, best_sig)

        return (text_sig, confidence)

    def _calculate_confidence(
        self, all_signatures: List[Dict], chosen_sig: Dict
    ) -> float:
        """Calculate confidence score for a chosen signature.

        Args:
            all_signatures: All available signatures
            chosen_sig: The chosen signature

        Returns:
            Confidence score from 0.0 to 1.0
        """
        if not all_signatures:
            return 0.0

        # Base confidence
        confidence = 0.5

        # Boost for standard signatures
        if chosen_sig.get("source") == "standard":
            confidence += 0.3

        # Boost if it's the only signature
        if len(all_signatures) == 1:
            confidence += 0.2

        # Penalty for many alternatives
        if len(all_signatures) > 3:
            confidence -= 0.1 * (len(all_signatures) - 3)
            confidence = max(confidence, 0.1)  # Minimum 0.1

        # Boost for recent signatures (from 4byte)
        created_at = chosen_sig.get("created_at", "")
        if created_at:
            try:
                # Parse ISO format
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                now_dt = datetime.now(created_dt.tzinfo)

                # More recent = higher confidence
                days_old = (now_dt - created_dt).days
                if days_old < 30:  # Less than 30 days old
                    confidence += 0.1
                elif days_old < 365:  # Less than 1 year old
                    confidence += 0.05
            except Exception:
                pass

        # Cap at 1.0
        return min(max(confidence, 0.0), 1.0)

    def analyze_selector(self, selector: str) -> Dict:
        """Comprehensive analysis of a selector.

        Args:
            selector: Function selector (hex string without 0x)

        Returns:
            Dictionary with analysis results
        """
        signatures = self.lookup_signatures(selector)

        if not signatures:
            return {
                "selector": selector,
                "status": "unknown",
                "signatures": [],
                "best_signature": None,
                "confidence": 0.0,
                "notes": "No signatures found in any database",
            }

        # Get best signature
        best_result = self.get_best_signature(selector)
        if best_result:
            best_sig, confidence = best_result
        else:
            best_sig, confidence = None, 0.0

        # Determine status
        if len(signatures) == 1:
            status = "unique"
        elif any(sig.get("source") == "standard" for sig in signatures):
            status = "standard"
        else:
            status = "ambiguous"

        # Extract just the text signatures for summary
        text_signatures = [sig.get("text_signature", "") for sig in signatures]

        return {
            "selector": selector,
            "status": status,
            "signatures": text_signatures,
            "signature_details": signatures,
            "best_signature": best_sig,
            "confidence": confidence,
            "total_signatures": len(signatures),
            "has_standard": any(sig.get("source") == "standard" for sig in signatures),
        }

    def batch_lookup(self, selectors: List[str]) -> Dict[str, Dict]:
        """Look up multiple selectors with rate limiting.

        Args:
            selectors: List of function selectors

        Returns:
            Dictionary mapping selector to analysis results
        """
        results = {}

        for i, selector in enumerate(selectors):
            logger.info(f"  Looking up 0x{selector} ({i + 1}/{len(selectors)})...")
            results[selector] = self.analyze_selector(selector)

            # Small delay between requests to be nice to the API
            if i < len(selectors) - 1:
                time.sleep(0.5)

        return results


# Test function
def test_signature_lookup():
    """Test the signature lookup."""
    logger.info("Testing SignatureLookup...")
    logger.info("=" * 60)

    lookup = SignatureLookup()

    # Test with known selectors
    test_selectors = [
        "a9059cbb",  # transfer (multiple signatures)
        "095ea7b3",  # approve (standard)
        "deadbeef",  # Unknown (should return empty)
        "70a08231",  # balanceOf (standard)
    ]

    for selector in test_selectors:
        logger.info(f"\nAnalyzing 0x{selector}:")

        analysis = lookup.analyze_selector(selector)

        logger.info(f"  Status: {analysis['status']}")
        logger.info(f"  Total signatures: {analysis['total_signatures']}")
        logger.info(f"  Has standard: {analysis['has_standard']}")

        if analysis["best_signature"]:
            logger.info(f"  Best signature: {analysis['best_signature']}")
            logger.info(f"  Confidence: {analysis['confidence']:.2f}")

        if analysis["signatures"]:
            logger.info("  All signatures:")
            for i, sig in enumerate(analysis["signatures"][:3]):  # Show first 3
                logger.info(f"    {i + 1}. {sig}")
            if len(analysis["signatures"]) > 3:
                logger.info(f"    ... and {len(analysis['signatures']) - 3} more")

    # Test batch lookup
    logger.info("\n\nBatch lookup test:")
    batch_results = lookup.batch_lookup(test_selectors[:2])

    for selector, result in batch_results.items():
        logger.info(f"  0x{selector}: {result['status']} ({result['total_signatures']} sigs)")

    return lookup


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_signature_lookup()
