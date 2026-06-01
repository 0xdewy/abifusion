#!/usr/bin/env python3
"""Plan #05 — train the ParameterPredictionModel on real verified contracts.

End-to-end deliverable for docs/plans/05-ml-training-pipeline.md:

  1. Fetch verified contracts from Etherscan (ABI + runtime bytecode).
  2. Write them to a parquet the training pipeline can consume.
  3. Train ParameterPredictionModel with discriminating features (CNN encoder).
  4. Evaluate on a held-out test split.

Discriminating features (SigRec R11–R18) are already wired through
ParameterDataset → ParameterPredictionModel → the trainer, so the existing
``abi_reconstructor.training.train_ml_models`` entry point trains *with* them;
this script supplies the missing piece — a real dataset — and orchestrates the
run by reusing existing components rather than reimplementing them.

Prerequisites for a real run:
  * ETHERSCAN_API_KEY in .env (loaded via python-dotenv; never printed).
  * Network access to api.etherscan.io (V2 endpoint).

Examples:
  # Validate wiring without network or training:
  python scripts/training/train_with_discriminating_features.py --dry-run

  # Small real run (needs a valid API key):
  python scripts/training/train_with_discriminating_features.py \
      --max-contracts 20 --epochs 3

  # Plan #05 target scale (discover contracts from Sourcify + ablation):
  python scripts/training/train_with_discriminating_features.py \
      --num-contracts 500 --epochs 15 --ablation --checkpoint-dir checkpoints
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("plan05")

# A seed set of well-known verified mainnet contracts spanning tokens, routers,
# pools and proxies — enough for a smoke run. Scale up via --addresses-file.
DEFAULT_ADDRESSES = [
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
    "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
    "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
    "0x514910771AF9Ca656af840dff83E8264EcF986CA",  # LINK
    "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI
    "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",  # AAVE
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2 Router
    "0xE592427A0AEce92De3Edee1F18E0157C05861564",  # Uniswap V3 Router
    "0x1111111254EEB25477B68fb85Ed929f73A960582",  # 1inch V4 Router
    "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",  # SushiSwap Router
    "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",  # Curve 3pool
    "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",  # Lido stETH
    "0xd9Db270c1B5E3Bd161E8c8503c55cEABeE709552",  # Gnosis Safe singleton
    "0xBA12222222228d8Ba445958a75a0704d566BF2C8",  # Balancer Vault
    "0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e",  # ENS Registry
    "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643",  # Compound cDAI
    "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",  # Maker (MKR)
    "0x111111111117dC0aa78b770fA6A738034120C302",  # 1INCH token
]


SOURCIFY_V2_CONTRACTS = "https://sourcify.dev/server/v2/contracts/{chain}"


def discover_addresses_from_sourcify(n: int, chain_id: int = 1) -> list:
    """Collect up to `n` verified contract addresses from Sourcify (keyless).

    Pages the Sourcify v2 contracts endpoint with afterMatchId until enough
    unique addresses are gathered or the listing is exhausted.
    """
    import requests

    url = SOURCIFY_V2_CONTRACTS.format(chain=chain_id)
    seen: list = []
    seen_set: set = set()
    after_match_id = None
    while len(seen) < n:
        params = {"limit": 200, "sort": "desc"}
        if after_match_id:
            params["afterMatchId"] = after_match_id
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for r in results:
            addr = r.get("address")
            if addr and addr.lower() not in seen_set:
                seen_set.add(addr.lower())
                seen.append(addr)
        after_match_id = results[-1].get("matchId")
        if not after_match_id:
            break
        logger.info("Sourcify discovery: %d/%d addresses", len(seen), n)
    return seen[:n]


def load_addresses(args) -> list:
    if args.addresses_file:
        text = Path(args.addresses_file).read_text()
        addrs = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    elif args.num_contracts > 0:
        # Over-collect: some Sourcify-verified contracts aren't Etherscan-verified
        # and get skipped during fetch.
        addrs = discover_addresses_from_sourcify(int(args.num_contracts * 1.6))
    else:
        addrs = list(DEFAULT_ADDRESSES)
    if args.max_contracts > 0:
        addrs = addrs[: args.max_contracts]
    return addrs


def fetch_dataset(addresses: list, data_dir: Path, target: int = 0) -> Path:
    """Fetch contracts from Etherscan and write a training parquet.

    Stops early once `target` contracts are successfully fetched (0 = no limit).
    """
    import pandas as pd
    from dotenv import load_dotenv

    from abi_reconstructor.data.etherscan_fetcher import EtherscanClient

    load_dotenv()  # populates ETHERSCAN_API_KEY from .env (not read by us)
    client = EtherscanClient()

    rows = []
    for i, address in enumerate(addresses):
        if target and len(rows) >= target:
            break
        logger.info("Fetching %d/%d (kept %d): %s", i + 1, len(addresses), len(rows), address)
        try:
            contract = client.get_contract(address, include_source_code=True)
        except Exception as e:  # noqa: BLE001 - keep fetching on per-contract errors
            logger.warning("  failed: %s", e)
            continue
        if not contract:
            continue
        rows.append(
            {
                "address": contract.address,
                "bytecode": contract.bytecode,
                "abi": contract.abi,
                "compiler_version": contract.compiler_version or "",
            }
        )

    if not rows:
        raise SystemExit(
            "No contracts fetched. Check ETHERSCAN_API_KEY in .env and network access."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "contracts.parquet"
    pd.DataFrame(rows).to_parquet(out)
    logger.info("Wrote %d contracts to %s", len(rows), out)
    return out


def run_training(args, data_dir: Path) -> int:
    """Reuse the existing training entry point (trains with discriminating features).

    Trains both the function-name classifier and the parameter model, then runs
    the full-pipeline evaluation (--evaluate). Because both models are trained
    in-process, --evaluate uses the freshly trained models and does not hit any
    stale on-disk checkpoint. Either encoder ("transformer" or "cnn") embeds the
    dataset's integer token ids directly.
    """
    cmd = [
        sys.executable,
        "-m",
        "abi_reconstructor.training.train_ml_models",
        "--data_dir", str(data_dir),
        "--checkpoint_dir", args.checkpoint_dir,
        "--train_function",
        "--train_parameter",
        "--evaluate",
        "--encoder_type", args.encoder_type,
        "--hidden_dim", "256",
        "--max_parameters", "12",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    logger.info("Training: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _type_accuracy(model, loader, zero_discriminating: bool) -> float:
    """Type accuracy over a loader, optionally zeroing discriminating features."""
    import torch

    correct = total = 0
    for batch in loader:
        disc = batch.get("discriminating_features")
        if zero_discriminating and disc is not None:
            disc = torch.zeros_like(disc)
        out = model.predict(
            batch["bytecode_tokens"],
            batch.get("attention_mask"),
            type_threshold=0.0,
            mask_threshold=0.5,
            discriminating_features=disc,
        )
        pred_types = out["raw_outputs"]["predicted_types"].cpu()
        true_types = batch["parameter_types"]
        true_mask = batch["parameter_mask"]
        max_params = pred_types.shape[1]
        for i in range(pred_types.shape[0]):
            for pos in range(max_params):
                if pos < true_mask.shape[1] and true_mask[i, pos].item() > 0.5:
                    total += 1
                    if int(pred_types[i, pos]) == int(true_types[i, pos]):
                        correct += 1
    return correct / total if total else 0.0


def run_ablation(args, data_dir: Path) -> None:
    """Compare test type-accuracy with real vs zeroed discriminating features."""
    from abi_reconstructor.models.parameter_prediction_model import (
        ParameterPredictionModel,
    )
    from abi_reconstructor.training.train_ml_models import prepare_datasets

    ckpt = Path(args.checkpoint_dir) / "parameter_predictor_final.pth"
    if not ckpt.exists():
        logger.warning("Ablation skipped: %s not found", ckpt)
        return

    _, _, test_loader, _, _ = prepare_datasets(
        data_dir=str(data_dir), batch_size=args.batch_size, seed=args.seed
    )
    model = ParameterPredictionModel.load_model(str(ckpt), use_cuda=True)

    acc_with = _type_accuracy(model, test_loader, zero_discriminating=False)
    acc_without = _type_accuracy(model, test_loader, zero_discriminating=True)
    logger.info("=== Discriminating-features ablation (test type accuracy) ===")
    logger.info("  with features:    %.4f", acc_with)
    logger.info("  zeroed features:  %.4f", acc_without)
    logger.info("  degradation:      %.4f", acc_with - acc_without)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--addresses-file", help="File with one contract address per line")
    parser.add_argument(
        "--num-contracts",
        type=int,
        default=0,
        help="Discover this many verified contracts from Sourcify (0 = use the "
        "built-in default set or --addresses-file)",
    )
    parser.add_argument("--max-contracts", type=int, default=0, help="Cap addresses (0 = all)")
    parser.add_argument("--data-dir", default="data", help="Where to write the dataset parquet")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/plan05",
        help="Checkpoint output dir (default keeps Plan #05 runs out of the "
        "canonical checkpoints/ to avoid clobbering existing models)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoder-type",
        default="transformer",
        choices=["transformer", "cnn"],
        help="Encoder for the parameter model (default: transformer; see run_training)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse an existing dataset parquet in --data-dir instead of fetching",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="After training, compare test type-accuracy with real vs zeroed "
        "discriminating features to confirm they are used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate wiring (imports, addresses, command) without network or training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    addresses = load_addresses(args)
    data_dir = Path(args.data_dir)
    logger.info("Plan #05 run: %d candidate contracts, %d epochs", len(addresses), args.epochs)

    if args.dry_run:
        # Import the pieces to prove the pipeline is wired, but do nothing costly.
        from abi_reconstructor.data.etherscan_fetcher import (
            EtherscanClient,  # noqa: F401
        )
        from abi_reconstructor.training.train_ml_models import (  # noqa: F401
            ParameterPredictionTrainer,
        )

        logger.info("Dry run OK: %d addresses, training reuses train_ml_models", len(addresses))
        return 0

    if not args.skip_fetch:
        fetch_dataset(addresses, data_dir, target=args.num_contracts)

    rc = run_training(args, data_dir)
    if rc != 0:
        logger.error("Training exited with code %d", rc)
        return rc

    if args.ablation:
        run_ablation(args, data_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
