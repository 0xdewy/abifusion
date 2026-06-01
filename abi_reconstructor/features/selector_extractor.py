"""Neural selector extractor for bytecode analysis.

This is the learned (CNN + attention) selector detector. For the lightweight,
pattern-based public utility see ``abi_reconstructor.selector_extractor``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from abi_reconstructor.device import get_device_for_cuda_flag, to_device


class NeuralSelectorExtractor(nn.Module):
    """Extract function selectors from bytecode using a learned CNN+attention model."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_cuda: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_cuda = use_cuda

        if device is None:
            device = get_device_for_cuda_flag(use_cuda)
        self._device = device

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Convolutional layers for pattern detection
        self.conv1 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)

        # Attention mechanism for selector detection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Selector classifier
        self.selector_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Binary classification: selector or not
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
            Dictionary with selector predictions and attention weights
        """
        # Move inputs to model's device
        input_ids = to_device(input_ids, self._device)
        if attention_mask is not None:
            attention_mask = to_device(attention_mask, self._device)

        batch_size, seq_len = input_ids.shape

        # Embed tokens
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, hidden_dim)

        # Apply convolutional layers
        # Transpose for conv1d: (batch_size, hidden_dim, seq_len)
        conv_input = embedded.transpose(1, 2)

        conv1_out = F.relu(self.conv1(conv_input))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))

        # Transpose back: (batch_size, seq_len, hidden_dim)
        conv_features = conv3_out.transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.attention(
            query=conv_features,
            key=conv_features,
            value=conv_features,
            key_padding_mask=attention_mask == 0
            if attention_mask is not None
            else None,
        )

        # Apply dropout
        attn_output = self.dropout_layer(attn_output)

        # Classify each position as selector or not
        selector_logits = self.selector_classifier(attn_output).squeeze(
            -1
        )  # (batch_size, seq_len)

        return {
            "selector_logits": selector_logits,
            "attention_weights": attn_weights,
            "conv_features": conv_features,
            "attention_output": attn_output,
        }

    def extract_selectors(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Extract selector positions from bytecode.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            threshold: Confidence threshold for selector detection

        Returns:
            Dictionary with selector positions and confidences
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            selector_logits = outputs["selector_logits"]
            selector_probs = torch.sigmoid(selector_logits)

            # Get selector positions above threshold
            selector_mask = selector_probs > threshold

            # Extract positions
            batch_selectors = []
            batch_confidences = []

            for i in range(selector_mask.size(0)):
                positions = torch.where(selector_mask[i])[0].cpu().numpy()
                confidences = selector_probs[i][selector_mask[i]].cpu().numpy()
                batch_selectors.append(positions)
                batch_confidences.append(confidences)

        return {
            "selector_positions": batch_selectors,
            "selector_confidences": batch_confidences,
            "selector_probs": selector_probs.cpu().numpy(),
            "selector_mask": selector_mask.cpu().numpy(),
        }
