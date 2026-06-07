import torch
from abi_reconstructor.ml.model import MLAbiModel
from abi_reconstructor.ml.dataset import load_contracts, ContractDataset, build_vocabs, prepare_test_set, compute_selector, sig_from_abi_func
from collections import Counter

df = load_contracts()
train_df = df.iloc[:400]
train_dataset = ContractDataset(train_df, {}, max_params=16)
name_vocab, num_names = build_vocabs(train_dataset.samples)
train_dataset.name_vocab = name_vocab

selector_counts = Counter()
for idx in range(len(df)):
    row = df.iloc[idx]
    for f in row['abi']:
        if f.get('type') == 'function':
            sel = compute_selector(sig_from_abi_func(f))
            selector_counts[sel] += 1

test_selectors = set()
for sel, count in selector_counts.items():
    if count == 1:
        for idx in range(450, 500):
            row = df.iloc[idx]
            for f in row['abi']:
                if f.get('type') == 'function' and compute_selector(sig_from_abi_func(f)) == sel:
                    test_selectors.add(sel)
                    break

test_df = df.iloc[450:500]
test_set = prepare_test_set(test_df, name_vocab, test_selectors, max_params=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLAbiModel(num_name_classes=num_names, num_type_classes=4, embed_dim=64, num_filters=128, max_params=16).to(device)
model.load_state_dict(torch.load('cache/ml_model.pt', map_location=device))
model.eval()

unk_idx = name_vocab.get('<unk>', 0)
unk_correct = 0
unk_total = 0
known_correct = 0
known_total = 0
known_top3_correct = 0

for s in test_set:
    fm = s.feature_matrix[:16]
    while len(fm) < 16:
        fm.append([0.0] * 8)

    tokens = s.bytecode_tokens[:5000]
    while len(tokens) < 5000:
        tokens.append(0)

    feature_tensor = torch.tensor([fm], dtype=torch.float32).to(device)
    token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    name_logits, _ = model(feature_tensor, token_tensor)
    label = name_vocab.get(s.name, unk_idx)
    pred = name_logits.argmax(dim=-1).item()
    top3 = name_logits.topk(3, dim=-1).indices[0].tolist()

    if label == unk_idx:
        unk_total += 1
        if pred == unk_idx:
            unk_correct += 1
    else:
        known_total += 1
        if pred == label:
            known_correct += 1
        if label in top3:
            known_top3_correct += 1

print(f'<unk> samples: {unk_total}, correct: {unk_correct}/{unk_total}')
print(f'Known-name samples: {known_total}, top-1 correct: {known_correct}/{known_total}')
print(f'Known-name Top-3 correct: {known_top3_correct}/{known_total}')
if known_total > 0:
    print(f'Known-name Top-3 accuracy: {known_top3_correct/known_total:.3f}')
