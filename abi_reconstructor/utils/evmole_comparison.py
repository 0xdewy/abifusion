"""EVMole comparison module for validating selector extraction."""

from typing import Dict, List, Set, Any, Optional
import json


def extract_selectors_with_evmole(bytecode: str) -> List[str]:
    """Extract function selectors using evmole library.

    Args:
        bytecode: Contract bytecode (hex string)

    Returns:
        List of selector strings (8 hex characters)
    """
    try:
        from evmole import contract_info

        # Clean bytecode (remove 0x prefix, whitespace)
        bytecode = bytecode.strip()
        if bytecode.startswith("0x"):
            bytecode = bytecode[2:]
        bytecode = bytecode.replace(" ", "").replace("\n", "").replace("\t", "")

        # Extract selectors using evmole
        info = contract_info(bytecode, selectors=True)

        if info.functions:
            return [func.selector for func in info.functions]
        return []

    except ImportError:
        print("Warning: evmole not installed. Install with: pip install evmole")
        return []
    except Exception as e:
        print(f"Error using evmole: {e}")
        return []


def compare_selectors(
    our_selectors: List[str],
    evmole_selectors: List[str],
    bytecode: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare selectors extracted by our method vs evmole.

    Args:
        our_selectors: Selectors found by our method
        evmole_selectors: Selectors found by evmole
        bytecode: Optional bytecode for debugging

    Returns:
        Dictionary with comparison results
    """
    our_set = set(our_selectors)
    evmole_set = set(evmole_selectors)

    common = our_set & evmole_set
    only_ours = our_set - evmole_set
    only_evmole = evmole_set - our_set

    # Calculate metrics
    precision = 0.0
    recall = 0.0
    f1_score = 0.0

    if our_set:
        precision = len(common) / len(our_set)

    if evmole_set:
        recall = len(common) / len(evmole_set)

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "our_selectors": sorted(list(our_set)),
        "evmole_selectors": sorted(list(evmole_set)),
        "common_selectors": sorted(list(common)),
        "only_our_selectors": sorted(list(only_ours)),
        "only_evmole_selectors": sorted(list(only_evmole)),
        "metrics": {
            "our_count": len(our_set),
            "evmole_count": len(evmole_set),
            "common_count": len(common),
            "only_our_count": len(only_ours),
            "only_evmole_count": len(only_evmole),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        },
        "bytecode_preview": bytecode[:100] + "..."
        if bytecode and len(bytecode) > 100
        else bytecode,
    }


def format_comparison_report(comparison: Dict[str, Any], verbose: bool = False) -> str:
    """Format comparison results as a readable report.

    Args:
        comparison: Comparison dictionary from compare_selectors
        verbose: Whether to include detailed selector lists

    Returns:
        Formatted report string
    """
    metrics = comparison["metrics"]

    report = []
    report.append("=" * 60)
    report.append("SELECTOR EXTRACTION COMPARISON REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"Our method found: {metrics['our_count']} selectors")
    report.append(f"EVMole found: {metrics['evmole_count']} selectors")
    report.append(f"Common selectors: {metrics['common_count']}")
    report.append(f"Only our method: {metrics['only_our_count']}")
    report.append(f"Only EVMole: {metrics['only_evmole_count']}")
    report.append("")
    report.append(f"Precision (our vs evmole): {metrics['precision']:.2%}")
    report.append(f"Recall (our vs evmole): {metrics['recall']:.2%}")
    report.append(f"F1 Score: {metrics['f1_score']:.2%}")
    report.append("")

    if verbose:
        report.append("Common selectors:")
        for selector in comparison["common_selectors"]:
            report.append(f"  - 0x{selector}")

        if comparison["only_our_selectors"]:
            report.append("")
            report.append("Only our method found:")
            for selector in comparison["only_our_selectors"]:
                report.append(f"  - 0x{selector}")

        if comparison["only_evmole_selectors"]:
            report.append("")
            report.append("Only EVMole found:")
            for selector in comparison["only_evmole_selectors"]:
                report.append(f"  - 0x{selector}")

    report.append("=" * 60)

    return "\n".join(report)


def save_comparison_report(comparison: Dict[str, Any], output_path: str) -> None:
    """Save comparison results to a JSON file.

    Args:
        comparison: Comparison dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"Comparison report saved to: {output_path}")


def batch_compare_selectors(
    bytecodes: List[str],
    our_extractor_func,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Batch compare selector extraction across multiple bytecodes.

    Args:
        bytecodes: List of bytecode strings
        our_extractor_func: Function that extracts selectors (takes bytecode, returns list)
        max_samples: Maximum number of samples to process

    Returns:
        Aggregate comparison statistics
    """
    if max_samples and max_samples < len(bytecodes):
        bytecodes = bytecodes[:max_samples]

    all_metrics = []

    for i, bytecode in enumerate(bytecodes):
        print(f"Processing {i + 1}/{len(bytecodes)}...")

        try:
            our_selectors = our_extractor_func(bytecode)
            evmole_selectors = extract_selectors_with_evmole(bytecode)

            comparison = compare_selectors(our_selectors, evmole_selectors, bytecode)
            all_metrics.append(comparison["metrics"])

        except Exception as e:
            print(f"Error processing bytecode {i + 1}: {e}")
            continue

    if not all_metrics:
        return {"error": "No successful comparisons"}

    # Calculate aggregate metrics
    total_our = sum(m["our_count"] for m in all_metrics)
    total_evmole = sum(m["evmole_count"] for m in all_metrics)
    total_common = sum(m["common_count"] for m in all_metrics)

    avg_precision = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["f1_score"] for m in all_metrics) / len(all_metrics)

    return {
        "sample_count": len(all_metrics),
        "total_our_selectors": total_our,
        "total_evmole_selectors": total_evmole,
        "total_common_selectors": total_common,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1,
        "per_sample_metrics": all_metrics,
    }
