"""Tests for signature_lookup.py."""

from unittest.mock import Mock, patch
import json
import os
import time
import hashlib

from abifusion.utils.signature_lookup import SignatureLookup


class TestSignatureLookup:
    """Test suite for SignatureLookup class."""

    def test_init(self, temp_cache_dir):
        """Test initialization with cache directory."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)
        assert lookup.cache_dir == temp_cache_dir
        assert lookup.session is not None
        assert hasattr(lookup, "last_request_time")
        assert lookup.min_request_interval == 1.0

    def test_init_default_cache(self):
        """Test initialization with default cache directory."""
        lookup = SignatureLookup()
        # Should use default cache location
        assert lookup.cache_dir is not None
        assert "cache" in lookup.cache_dir
        assert "signatures" in lookup.cache_dir

    def test_get_cache_key(self):
        """Test cache key generation."""
        lookup = SignatureLookup()
        assert lookup._get_cache_key("a9059cbb") == "sig_a9059cbb"
        assert lookup._get_cache_key("095ea7b3") == "sig_095ea7b3"

    def test_get_cache_path(self, temp_cache_dir):
        """Test cache path generation."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)
        cache_key = "sig_a9059cbb"
        cache_path = lookup._get_cache_path(cache_key)

        # Should be MD5 hash of cache key
        expected_hash = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
        assert cache_path.endswith(expected_hash)
        assert cache_path.startswith(temp_cache_dir)

    def test_load_from_cache_empty(self, temp_cache_dir):
        """Test loading from empty cache."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)
        result = lookup._load_from_cache("sig_a9059cbb")
        assert result is None

    def test_load_from_cache_existing(self, temp_cache_dir):
        """Test loading from existing cache."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)
        cache_key = "sig_a9059cbb"
        cache_path = lookup._get_cache_path(cache_key)

        # Create cache file
        test_data = {
            "timestamp": time.time(),
            "signatures": [
                {"text_signature": "transfer(address,uint256)", "priority": 1},
                {
                    "text_signature": "workMyDirefulOwner(uint256,uint256)",
                    "priority": 2,
                },
            ],
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(test_data, f)

        result = lookup._load_from_cache(cache_key)
        assert result is not None
        assert len(result) == 2
        assert result[0]["text_signature"] == "transfer(address,uint256)"
        assert result[1]["text_signature"] == "workMyDirefulOwner(uint256,uint256)"

    def test_load_from_cache_expired(self, temp_cache_dir):
        """Test loading from expired cache."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)
        cache_key = "sig_a9059cbb"
        cache_path = lookup._get_cache_path(cache_key)

        # Create expired cache file (31 days old)
        test_data = {
            "timestamp": time.time() - (31 * 24 * 3600),  # 31 days ago
            "signatures": [
                {"text_signature": "transfer(address,uint256)", "priority": 1}
            ],
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(test_data, f)

        result = lookup._load_from_cache(cache_key)
        assert result is None  # Should return None for expired cache

    def test_save_to_cache(self, temp_cache_dir):
        """Test saving to cache."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)
        cache_key = "sig_a9059cbb"
        test_signatures = [
            {"text_signature": "transfer(address,uint256)", "priority": 1},
            {"text_signature": "workMyDirefulOwner(uint256,uint256)", "priority": 2},
        ]

        lookup._save_to_cache(cache_key, test_signatures)

        cache_path = lookup._get_cache_path(cache_key)
        assert os.path.exists(cache_path)

        with open(cache_path, "r") as f:
            saved_data = json.load(f)

        assert "timestamp" in saved_data
        assert "signatures" in saved_data
        assert len(saved_data["signatures"]) == 2
        assert (
            saved_data["signatures"][0]["text_signature"] == "transfer(address,uint256)"
        )

    def test_lookup_signatures_cache_hit(self, temp_cache_dir):
        """Test lookup with cache hit."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)
        cache_key = "sig_a9059cbb"

        # Setup cache with data
        test_signatures = [
            {"text_signature": "transfer(address,uint256)", "priority": 1},
            {"text_signature": "workMyDirefulOwner(uint256,uint256)", "priority": 2},
        ]
        lookup._save_to_cache(cache_key, test_signatures)

        # Mock should NOT be called since we have cache hit
        with patch("requests.get") as mock_get:
            results = lookup.lookup_signatures("a9059cbb")

            # Verify API was not called
            mock_get.assert_not_called()

            # Verify we got cached results (plus standard signatures)
            # Results should include both cached and standard signatures
            assert len(results) >= 2

            # Check that our cached signatures are in results
            text_signatures = [r["text_signature"] for r in results]
            assert "transfer(address,uint256)" in text_signatures

    def test_lookup_signatures_cache_miss(self, temp_cache_dir, mock_4byte_response):
        """Test lookup with cache miss."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)

        # Use a selector that doesn't have standard signatures
        test_selector = "12345678"  # Not in STANDARD_SIGNATURES
        cache_key = lookup._get_cache_key(test_selector)
        cache_path = lookup._get_cache_path(cache_key)

        # Clear cache for this selector
        if os.path.exists(cache_path):
            os.remove(cache_path)

        # Setup mock response for session.get()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_4byte_response
        mock_response.raise_for_status.return_value = None

        # Mock the session.get() method
        with patch.object(lookup.session, "get") as mock_session_get:
            mock_session_get.return_value = mock_response

            results = lookup.lookup_signatures(test_selector)

            # Verify API was called (since no standard signatures and no cache)
            mock_session_get.assert_called_once()

            # Verify we got results
            assert len(results) > 0

            # Verify cache was updated
            assert os.path.exists(cache_path)
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            assert "signatures" in cache_data
            assert len(cache_data["signatures"]) > 0

    def test_lookup_signatures_api_error(self, temp_cache_dir):
        """Test lookup when API returns error."""
        import requests

        lookup = SignatureLookup(cache_dir=temp_cache_dir)

        # Use a selector with standard signatures
        test_selector = "a9059cbb"

        # Setup mock error response for session.get()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "API Error"
        )

        # Mock the session.get() method
        with patch.object(lookup.session, "get") as mock_session_get:
            mock_session_get.return_value = mock_response

            results = lookup.lookup_signatures(test_selector)

            # Should still return standard signatures even on API error
            assert len(results) > 0
            # Should include standard transfer signature
            text_signatures = [r["text_signature"] for r in results]
            assert "transfer(address,uint256)" in text_signatures

    def test_lookup_signatures_network_error(self, temp_cache_dir):
        """Test lookup when network request fails."""
        import requests

        lookup = SignatureLookup(cache_dir=temp_cache_dir)

        # Use a selector with standard signatures
        test_selector = "a9059cbb"

        # Mock the session.get() method to raise exception
        with patch.object(lookup.session, "get") as mock_session_get:
            mock_session_get.side_effect = requests.exceptions.RequestException(
                "Network error"
            )

            results = lookup.lookup_signatures(test_selector)

            # Should still return standard signatures even on network error
            assert len(results) > 0
            # Should include standard transfer signature
            text_signatures = [r["text_signature"] for r in results]
            assert "transfer(address,uint256)" in text_signatures

    def test_lookup_signatures_empty_response(self, temp_cache_dir):
        """Test lookup when API returns empty response."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)

        # Use a selector without standard signatures
        test_selector = "unknown1234"

        # Setup mock empty response for session.get()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 0, "results": []}
        mock_response.raise_for_status.return_value = None

        # Mock the session.get() method
        with patch.object(lookup.session, "get") as mock_session_get:
            mock_session_get.return_value = mock_response

            results = lookup.lookup_signatures(test_selector)

            # Should return empty list (no standard signatures for unknown selector)
            assert isinstance(results, list)

    def test_lookup_signatures_invalid_selector(self):
        """Test lookup with invalid selector."""
        lookup = SignatureLookup()

        # Empty selector - should handle gracefully
        results = lookup.lookup_signatures("")
        assert isinstance(results, list)

        # Too short selector
        results = lookup.lookup_signatures("abc")
        assert isinstance(results, list)

        # Non-hex selector
        results = lookup.lookup_signatures("nothex!!")
        assert isinstance(results, list)

    def test_get_standard_signatures(self):
        """Test getting standard signatures."""
        lookup = SignatureLookup()

        # Test getting standard signature for known selector
        results = lookup.get_standard_signatures("a9059cbb")
        assert len(results) == 1
        assert results[0] == "transfer(address,uint256)"

        # Test getting standard signature for another selector
        results = lookup.get_standard_signatures("095ea7b3")
        assert len(results) == 1
        assert results[0] == "approve(address,uint256)"

        # Test getting standard signature for unknown selector
        results = lookup.get_standard_signatures("unknown1234")
        assert results == []

    def test_rate_limiting(self, temp_cache_dir):
        """Test rate limiting between requests."""
        lookup = SignatureLookup(cache_dir=temp_cache_dir)

        # Use selectors without standard signatures
        test_selector1 = "11111111"
        test_selector2 = "22222222"

        # Clear cache for these selectors
        cache_key1 = lookup._get_cache_key(test_selector1)
        cache_path1 = lookup._get_cache_path(cache_key1)
        if os.path.exists(cache_path1):
            os.remove(cache_path1)

        cache_key2 = lookup._get_cache_key(test_selector2)
        cache_path2 = lookup._get_cache_path(cache_key2)
        if os.path.exists(cache_path2):
            os.remove(cache_path2)

        # Setup mock response for session.get()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "count": 1,
            "results": [{"text_signature": "test()"}],
        }
        mock_response.raise_for_status.return_value = None

        # Mock the session.get() method
        with patch.object(lookup.session, "get") as mock_session_get:
            mock_session_get.return_value = mock_response

            # First call
            lookup.lookup_signatures(test_selector1)

            # Reset mock to track second call
            mock_session_get.reset_mock()
            mock_session_get.return_value = mock_response

            # Second call immediately after should respect rate limit
            with patch("time.sleep") as mock_sleep:
                lookup.lookup_signatures(test_selector2)

                # Should have slept due to rate limiting
                mock_sleep.assert_called_once()
                # Should have made the request
                mock_session_get.assert_called_once()

    def test_standard_signatures_priority(self):
        """Test that standard signatures get priority."""
        lookup = SignatureLookup()

        # For a standard selector, standard signature should be first
        results = lookup.lookup_signatures("a9059cbb")

        # Find standard signature in results
        standard_found = False
        for sig in results:
            if sig.get("text_signature") == "transfer(address,uint256)":
                # Standard signature should have priority 1
                assert sig.get("priority") == 1
                standard_found = True
                break

        assert standard_found, "Standard signature not found in results"
