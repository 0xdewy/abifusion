"""Extract function selectors from EVM bytecode."""

from typing import List, Tuple

from .reconstructor import BytecodeParser, FunctionSelector


class SelectorExtractor:
    """Extract function selectors from bytecode using pattern matching."""

    def __init__(self, use_ml: bool = False):
        """Initialize selector extractor.

        Args:
            use_ml: Whether to use ML-based extraction (not yet implemented)
        """
        self.parser = BytecodeParser()
        self.use_ml = use_ml

    def extract(
        self,
        bytecode: str,
        max_selectors: int = 50
    ) -> List[FunctionSelector]:
        """Extract function selectors from bytecode.

        Args:
            bytecode: Contract bytecode (hex string)
            max_selectors: Maximum number of selectors to return

        Returns:
            List of FunctionSelector objects
        """
        selectors = self.parser.extract_selectors(bytecode)

        if max_selectors > 0 and len(selectors) > max_selectors:
            selectors = selectors[:max_selectors]

        return selectors

    def extract_with_signatures(
        self,
        bytecode: str,
        db,  # FourByteDatabase
        max_selectors: int = 50
    ) -> List[Tuple[FunctionSelector, str]]:
        """Extract selectors and lookup signatures.

        Args:
            bytecode: Contract bytecode
            db: FourByteDatabase instance
            max_selectors: Maximum number of selectors

        Returns:
            List of (selector, signature) tuples
        """
        selectors = self.extract(bytecode, max_selectors)
        results = []

        for sel in selectors:
            sigs = db.lookup(sel.selector, fetch_api=True)
            signature = sigs[0].text_signature if sigs else f"unknown_{sel.selector}()"
            results.append((sel, signature))

        return results

    def filter_by_confidence(
        self,
        selectors: List[FunctionSelector],
        min_confidence: float = 0.5
    ) -> List[FunctionSelector]:
        """Filter selectors by minimum confidence.

        Args:
            selectors: List of FunctionSelector objects
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of selectors
        """
        return [s for s in selectors if s.confidence >= min_confidence]

    def get_selector_positions(self, bytecode: str) -> List[int]:
        """Get all positions where selectors are found.

        Args:
            bytecode: Contract bytecode

        Returns:
            List of position indices
        """
        selectors = self.parser.extract_selectors(bytecode)
        return [s.position for s in selectors]