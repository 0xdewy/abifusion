"""ML model for zero-shot ABI reconstruction.

Architecture:
- Bytecode encoder: 1D CNN over tokenized bytecode bytes (0-255)
- Per-position fusion: [num_params × 8] SigRec features concatenated with bytecode embeddings
- Two heads:
  - Function name head: Top-3 classification over ~2000-name vocabulary
  - Parameter type head: Per-position 4-class (bytes-like, uint-like, address, other)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BytecodeEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, num_filters: int = 128, kernel_sizes: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(embedded))
            h, _ = h.max(dim=-1)
            conv_outputs.append(h)
        out = torch.cat(conv_outputs, dim=-1)
        out = self.fc(out)
        return F.relu(out)


class NameHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class TypeHead(nn.Module):
    def __init__(self, feature_dim: int = 8, embed_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(feature_dim + embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, features: torch.Tensor, bytecode_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([features, bytecode_embed], dim=-1)
        h = F.relu(self.fc(combined))
        return self.classifier(h)


class MLAbiModel(nn.Module):
    def __init__(
        self,
        num_name_classes: int,
        num_type_classes: int = 4,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        max_params: int = 16,
    ):
        super().__init__()
        self.max_params = max_params
        self.bytecode_encoder = BytecodeEncoder(
            vocab_size=256, embed_dim=embed_dim, num_filters=num_filters, kernel_sizes=kernel_sizes
        )
        self.name_head = NameHead(embed_dim, num_name_classes)
        self.type_head = TypeHead(feature_dim=9, embed_dim=embed_dim, num_classes=num_type_classes)

    def forward(
        self,
        feature_matrix: torch.Tensor,
        bytecode_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bytecode_embed = self.bytecode_encoder(bytecode_tokens)
        name_logits = self.name_head(bytecode_embed)

        batch_size = bytecode_embed.size(0)
        max_p = min(feature_matrix.size(1), self.max_params)
        bytecode_embed_expanded = bytecode_embed.unsqueeze(1).expand(batch_size, max_p, -1)
        combined = torch.cat([feature_matrix[:, :max_p], bytecode_embed_expanded], dim=-1)
        flat = combined.view(batch_size * max_p, -1)
        type_logits = self.type_head.fc(flat)
        type_logits = F.relu(type_logits)
        type_logits = self.type_head.classifier(type_logits)
        type_logits = type_logits.view(batch_size, max_p, -1)
        return name_logits, type_logits


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    preds = logits.topk(k, dim=-1).indices
    correct = 0
    for i, label in enumerate(labels):
        if label.item() in preds[i].tolist():
            correct += 1
    return correct / len(labels) if len(labels) > 0 else 0.0
