"""Train ML model for zero-shot ABI reconstruction using function family prediction.

Instead of predicting exact function names (1,656 classes), we predict function
families (~33 classes). This dramatically reduces the classification difficulty
and enables generalization to unseen function names.

Usage:
    python scripts/train/train_family_model.py
    python scripts/train/train_family_model.py --quick # fast dev run
    python scripts/train/train_family_model.py --test-only  # test existing model
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from abi_reconstructor.ml.dataset import (
    ContractDataset,
    compute_selector,
    load_contracts,
    prepare_test_set,
    sig_from_abi_func,
)
from abi_reconstructor.ml.families import FAMILY_NAMES, get_family
from abi_reconstructor.ml.model import MLAbiModel


def get_all_selectors(contracts_df):
    selector_counts = Counter()
    for idx in range(len(contracts_df)):
        row = contracts_df.iloc[idx]
        abi = row["abi"]
        funcs = [f for f in abi if f.get("type") == "function"]
        for f in funcs:
            sig = sig_from_abi_func(f)
            sel = compute_selector(sig)
            selector_counts[sel] += 1
    return selector_counts


def train_epoch(model, dataloader, optimizer, device, max_params):
    model.train()
    total_name_loss = 0.0
    total_type_loss = 0.0
    total_name_correct = 0
    total_name_total = 0
    total_type_correct = 0
    total_type_total = 0

    for batch in dataloader:
        feature_matrix, bytecode_tokens, name_labels, type_labels = batch
        feature_matrix = feature_matrix.to(device)
        bytecode_tokens = bytecode_tokens.to(device)
        name_labels = name_labels.to(device)
        type_labels = type_labels.to(device)

        optimizer.zero_grad()

        name_logits, type_logits = model(feature_matrix, bytecode_tokens)

        name_loss = F.cross_entropy(name_logits, name_labels)
        total_name_correct += (name_logits.argmax(dim=-1) == name_labels).sum().item()
        total_name_total += name_labels.size(0)

        batch_size = bytecode_tokens.size(0)
        num_params = min(feature_matrix.size(1), max_params)
        flat_type_labels = type_labels[:, :num_params].reshape(-1)
        flat_type_logits = type_logits[:, :num_params].reshape(batch_size * num_params, -1)
        valid_mask = flat_type_labels >= 0
        flat_type_logits = flat_type_logits[valid_mask]
        flat_labels = flat_type_labels[valid_mask]

        if flat_labels.numel() > 0:
            type_loss = F.cross_entropy(flat_type_logits, flat_labels)
            total_type_correct += (flat_type_logits.argmax(dim=-1) == flat_labels).sum().item()
            total_type_total += flat_labels.size(0)
        else:
            type_loss = torch.tensor(0.0, device=device)

        loss = name_loss + type_loss
        loss.backward()

        optimizer.step()

        total_name_loss += name_loss.item()
        total_type_loss += type_loss.item()

    n = len(dataloader)
    return {
        "name_loss": total_name_loss / n,
        "type_loss": total_type_loss / n,
        "name_acc": total_name_correct / total_name_total if total_name_total > 0 else 0,
        "type_acc": total_type_correct / total_type_total if total_type_total > 0 else 0,
    }


def evaluate(model, dataloader, device, max_params):
    model.eval()
    total_name_loss = 0.0
    total_name_correct = 0
    total_name_total = 0
    total_name_top3_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            feature_matrix, bytecode_tokens, name_labels, type_labels = batch
            feature_matrix = feature_matrix.to(device)
            bytecode_tokens = bytecode_tokens.to(device)
            name_labels = name_labels.to(device)

            name_logits, _ = model(feature_matrix, bytecode_tokens)
            name_loss = F.cross_entropy(name_logits, name_labels)
            total_name_loss += name_loss.item()

            preds = name_logits.argmax(dim=-1)
            total_name_correct += (preds == name_labels).sum().item()
            total_name_total += name_labels.size(0)

            top3_preds = name_logits.topk(3, dim=-1).indices
            for i, label in enumerate(name_labels):
                if label.item() in top3_preds[i].tolist():
                    total_name_top3_correct += 1

    n = len(dataloader)
    return {
        "name_loss": total_name_loss / n,
        "name_acc": total_name_correct / total_name_total if total_name_total > 0 else 0,
        "name_top3_acc": total_name_top3_correct / total_name_total if total_name_total > 0 else 0,
    }


def evaluate_test_set(model, test_samples, device, max_params, batch_size=16):
    model.eval()
    name_correct = 0
    name_top3_correct = 0
    type_correct = 0
    type_total = 0

    with torch.no_grad():
        for i in range(0, len(test_samples), batch_size):
            batch_samples = test_samples[i : i + batch_size]
            feature_matrices = []
            bytecode_tokens_list = []
            name_labels_list = []
            type_labels_list = []
            num_params_list = []

            for s in batch_samples:
                fm = s.feature_matrix[:max_params]
                while len(fm) < max_params:
                    fm.append([0.0] * 8)
                feature_matrices.append(fm)

                tokens = s.bytecode_tokens[:5000]
                while len(tokens) < 5000:
                    tokens.append(0)
                bytecode_tokens_list.append(tokens)

                name_labels_list.append(get_family(s.name))
                num_params_list.append(min(max_params, len(s.coarse_types)))
                ct = s.coarse_types[:max_params]
                while len(ct) < max_params:
                    ct.append(0)
                type_labels_list.append(ct[:max_params])

            feature_tensor = torch.tensor(feature_matrices, dtype=torch.float32).to(device)
            token_tensor = torch.tensor(bytecode_tokens_list, dtype=torch.long).to(device)
            name_labels = torch.tensor(name_labels_list, dtype=torch.long).to(device)

            name_logits, type_logits = model(feature_tensor, token_tensor)

            preds = name_logits.argmax(dim=-1)
            name_correct += (preds == name_labels).sum().item()

            top3_preds = name_logits.topk(3, dim=-1).indices
            for j, label in enumerate(name_labels):
                if label.item() in top3_preds[j].tolist():
                    name_top3_correct += 1

            for b, num_p in enumerate(num_params_list):
                if num_p > 0:
                    type_logits_pos = type_logits[b, :num_p]
                    type_labels_pos = torch.tensor(type_labels_list[b][:num_p], dtype=torch.long).to(device)
                    type_correct += (type_logits_pos.argmax(dim=-1) == type_labels_pos).sum().item()
                    type_total += num_p

    total = len(test_samples)
    return {
        "name_acc": name_correct / total if total > 0 else 0,
        "name_top3_acc": name_top3_correct / total if total > 0 else 0,
        "type_acc": type_correct / type_total if type_total > 0 else 0,
        "type_total_positions": type_total,
        "num_samples": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast dev run with small model")
    parser.add_argument("--test-only", action="store_true", help="Test existing model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--max-params", type=int, default=16)
    parser.add_argument("--model-path", type=str, default="cache/family_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Family classes: {len(FAMILY_NAMES)}")

    print("Loading contracts...")
    df = load_contracts()

    print("Computing selector frequencies...")
    selector_counts = get_all_selectors(df)
    unique_selectors = {sel for sel, count in selector_counts.items() if count == 1}
    print(f"Unique selectors in dataset: {len(unique_selectors)}")

    test_selectors_in_test_contracts = set()
    for sel, count in selector_counts.items():
        if count == 1:
            for idx in range(len(df)):
                row = df.iloc[idx]
                abi = row["abi"]
                funcs = [f for f in abi if f.get("type") == "function"]
                for f in funcs:
                    sig = sig_from_abi_func(f)
                    if compute_selector(sig) == sel and idx >= 450:
                        test_selectors_in_test_contracts.add(sel)
                        break

    print(f"Test selectors (unique, in test contracts 450-500): {len(test_selectors_in_test_contracts)}")

    train_df = df.iloc[:400]
    val_df = df.iloc[400:450]
    test_df = df.iloc[450:500]

    print(f"Train contracts: {len(train_df)}, Val contracts: {len(val_df)}, Test contracts: {len(test_df)}")

    print("Building training dataset (families)...")
    train_dataset = ContractDataset(train_df, {}, max_params=args.max_params, use_families=True)
    val_dataset = ContractDataset(val_df, {}, max_params=args.max_params, use_families=True)

    print(f"Training samples: {len(train_dataset.samples)}, Family classes: {len(FAMILY_NAMES)}")

    if args.quick:
        args.epochs = 3
        args.batch_size = 8
        args.embed_dim = 32
        args.num_filters = 64

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_family_classes = len(FAMILY_NAMES)
    model = MLAbiModel(
        num_name_classes=num_family_classes,
        num_type_classes=4,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        max_params=args.max_params,
    ).to(device)

    if args.test_only:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        test_set = prepare_test_set(test_df, {}, test_selectors_in_test_contracts, max_params=args.max_params)
        results = evaluate_test_set(model, test_set, device, args.max_params)
        print(f"\nTest set results: {results}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path("cache").mkdir(exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.max_params)
        val_metrics = evaluate(model, val_loader, device, args.max_params)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) "
            f"train_name_loss={train_metrics['name_loss']:.3f} "
            f"train_name_acc={train_metrics['name_acc']:.3f} "
            f"val_name_loss={val_metrics['name_loss']:.3f} "
            f"val_name_acc={val_metrics['name_acc']:.3f} "
            f"val_name_top3_acc={val_metrics['name_top3_acc']:.3f}"
        )

        if val_metrics["name_top3_acc"] > best_val_acc:
            best_val_acc = val_metrics["name_top3_acc"]
            torch.save(model.state_dict(), args.model_path)
            print(f"  -> Saved best model (val_top3_acc={best_val_acc:.3f})")

    print("\nEvaluating on zero-shot test set...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    test_set = prepare_test_set(test_df, {}, test_selectors_in_test_contracts, max_params=args.max_params)
    results = evaluate_test_set(model, test_set, device, args.max_params)
    print("\nTest set results:")
    print(f"  Family Top-3 accuracy: {results['name_top3_acc']:.3f} ({results['name_acc']:.3f} top-1)")
    print(f"  Type accuracy: {results['type_acc']:.3f} ({results['type_total_positions']} positions)")
    print(f"  Num samples: {results['num_samples']}")

    threshold = 0.60
    if results["name_top3_acc"] >= threshold:
        print(f"\nPASS: {results['name_top3_acc']:.1%} >= {threshold:.0%} threshold — integrate ML model")
    else:
        print(f"\nFAIL: {results['name_top3_acc']:.1%} < {threshold:.0%} threshold — do not integrate")


if __name__ == "__main__":
    main()
