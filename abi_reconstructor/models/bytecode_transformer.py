"""Bytecode transformer and CNN encoder models."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from abi_reconstructor.device import get_device_for_cuda_flag, to_device


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, device: Optional[torch.device] = None):
        super().__init__()
        self.d_model = d_model

        if device is None:
            device = torch.device("cpu")

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class BytecodeTransformer(nn.Module):
    """Transformer model for bytecode sequence processing."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_positional_encoding: bool = True,
        use_cuda: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_positional_encoding = use_positional_encoding
        self.use_cuda = use_cuda

        if device is None:
            device = get_device_for_cuda_flag(use_cuda)
        self._device = device

        self.embedding = nn.Embedding(vocab_size, d_model)

        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, device=self._device)
        else:
            self.pos_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout_layer = nn.Dropout(dropout)

        if use_cuda:
            self.to(self._device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = to_device(input_ids, self._device)
        if attention_mask is not None:
            attention_mask = to_device(attention_mask, self._device)

        batch_size, seq_len = input_ids.shape

        if batch_size == 0:
            return {
                "last_hidden_state": torch.zeros(0, seq_len, self.d_model, device=self._device),
                "pooled_output": torch.zeros(0, self.d_model, device=self._device),
                "attention_mask": attention_mask,
            }

        x = self.embedding(input_ids)
        x = x * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            actual_seq_len = x.size(1)
            if actual_seq_len > self.max_seq_len:
                x = x[:, :self.max_seq_len, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.max_seq_len]
            x = self.pos_encoder(x)

        x = self.dropout_layer(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask_float = attention_mask.float()
                src_key_padding_mask = attention_mask_float == 0
            else:
                src_key_padding_mask = ~attention_mask

        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None and src_key_padding_mask is not None:
            mask_expanded = (~src_key_padding_mask).float().unsqueeze(-1)
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        return {
            "last_hidden_state": encoded,
            "pooled_output": pooled,
            "attention_mask": attention_mask,
        }

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input to get pooled representation."""
        outputs = self.forward(input_ids, attention_mask)
        return outputs["pooled_output"]


class BytecodeCNNEncoder(nn.Module):
    """CNN encoder for bytecode sequences."""

    def __init__(
        self,
        vocab_size: int = 256,
        embedding_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: Tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.2,
        use_cuda: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.output_dim = num_filters * len(filter_sizes)

        if device is None:
            device = get_device_for_cuda_flag(use_cuda)
        self._device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                    padding=fs // 2,
                )
                for fs in filter_sizes
            ]
        )

        self.dropout_layer = nn.Dropout(dropout)

        if use_cuda:
            self.to(self._device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = to_device(input_ids, self._device)
        if attention_mask is not None:
            attention_mask = to_device(attention_mask, self._device)

        batch_size, seq_len = input_ids.shape

        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)
            conv_out = F.relu(conv_out)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        encoded = torch.cat(conv_outputs, dim=1)
        encoded = self.dropout_layer(encoded)

        return {
            "last_hidden_state": encoded,
            "pooled_output": encoded,
            "attention_mask": attention_mask,
        }

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input to get representation."""
        outputs = self.forward(input_ids, attention_mask)
        return outputs["last_hidden_state"]