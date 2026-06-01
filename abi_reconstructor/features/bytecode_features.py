"""Bytecode feature extractor."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from abi_reconstructor.device import get_device_for_cuda_flag, to_device


class BytecodeFeatureExtractor(nn.Module):
    """Extract features from bytecode sequences."""

    def __init__(
        self,
        vocab_size: int = 256,
        feature_dim: int = 256,
        dropout: float = 0.2,
        use_cuda: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.use_cuda = use_cuda

        if device is None:
            device = get_device_for_cuda_flag(use_cuda)
        self._device = device

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, feature_dim)

        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Move to device if specified and available
        if use_cuda:
            self.to(self._device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Dictionary with extracted features
        """
        # Move inputs to model's device
        input_ids = to_device(input_ids, self._device)
        if attention_mask is not None:
            attention_mask = to_device(attention_mask, self._device)

        batch_size, seq_len = input_ids.shape

        # Handle empty batch case
        if batch_size == 0:
            return {
                "pooled_features": torch.zeros(0, self.feature_dim, device=self._device),
                "sequence_features": torch.zeros(0, seq_len, self.feature_dim, device=self._device),
            }

        # Embed tokens
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, feature_dim)

        # Apply convolutional layers
        # Transpose for conv1d: (batch_size, feature_dim, seq_len)
        conv_input = embedded.transpose(1, 2)
        conv_features = self.conv_layers(
            conv_input
        )  # (batch_size, feature_dim, seq_len)

        # Transpose back: (batch_size, seq_len, feature_dim)
        conv_features = conv_features.transpose(1, 2)

        # Apply dropout
        conv_features = self.dropout_layer(conv_features)

        # Attention pooling
        attention_scores = self.attention_pool(
            conv_features
        )  # (batch_size, seq_len, 1)

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum of features
        pooled_features = torch.sum(
            conv_features * attention_weights, dim=1
        )  # (batch_size, feature_dim)

        return {
            "sequence_features": conv_features,  # (batch_size, seq_len, feature_dim)
            "pooled_features": pooled_features,  # (batch_size, feature_dim)
            "attention_weights": attention_weights,  # (batch_size, seq_len, 1)
        }

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Extract features from bytecode.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            return_sequence: If True, return sequence features; if False, return pooled features

        Returns:
            Extracted features
        """
        outputs = self.forward(input_ids, attention_mask)
        if return_sequence:
            return outputs["sequence_features"]
        else:
            return outputs["pooled_features"]

    def extract_basic_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for extract_features for backward compatibility."""
        return self.extract_features(input_ids, attention_mask, return_sequence=False)

    def get_attention_patterns(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Get attention patterns for interpretability.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            Dictionary with attention patterns
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

            attention_weights = outputs["attention_weights"].squeeze(
                -1
            )  # (batch_size, seq_len)

            # Get top attention positions
            batch_size, seq_len = attention_weights.shape
            top_k = min(5, seq_len)

            top_weights, top_indices = torch.topk(attention_weights, k=top_k, dim=1)

            batch_patterns = []
            for i in range(batch_size):
                patterns = {
                    "positions": top_indices[i].cpu().numpy(),
                    "weights": top_weights[i].cpu().numpy(),
                }
                batch_patterns.append(patterns)

        return {
            "attention_patterns": batch_patterns,
            "attention_weights": attention_weights.cpu().numpy(),
        }
