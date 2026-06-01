"""Function name classifier model for ABI reconstruction."""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FunctionNameClassifier(nn.Module):
    """Classifier for predicting function names from bytecode features."""

    def __init__(
        self,
        num_classes: int,
        encoder_type: str = "transformer",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_cuda: bool = True,
        encoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_cuda = use_cuda

        # Default encoder config
        if encoder_config is None:
            encoder_config = {
                "vocab_size": 257,
                "d_model": hidden_dim,
                "nhead": 8,
                "num_layers": 3,
                "dim_feedforward": 512,
                "dropout": dropout,
            }

        self.encoder_config = encoder_config

        # Create encoder based on type
        if encoder_type == "transformer":
            self.encoder = self._create_transformer_encoder(encoder_config)
        elif encoder_type == "cnn":
            self.encoder = self._create_cnn_encoder(encoder_config)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # Projection from encoder output to hidden_dim. The transformer outputs
        # d_model; the CNN encoder always outputs hidden_dim.
        if encoder_type == "cnn":
            encoder_out_dim = hidden_dim
        else:
            encoder_out_dim = encoder_config.get("d_model", hidden_dim)
        self.encoder_projection = nn.Linear(encoder_out_dim, hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Move to device through DeviceManager
        from abi_reconstructor.device import get_device_for_cuda_flag
        device_obj = get_device_for_cuda_flag(use_cuda)
        if device_obj.type != "cpu":
            self.to(device_obj)

    def _create_transformer_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Create transformer encoder."""
        # Create embedding layer
        vocab_size = config.get("vocab_size", 256)
        d_model = config["d_model"]
        embedding = nn.Embedding(vocab_size, d_model)

        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )

        # Combine embedding and transformer
        class TransformerWithEmbedding(nn.Module):
            def __init__(self, embedding, transformer):
                super().__init__()
                self.embedding = embedding
                self.transformer = transformer

            def forward(self, input_ids, src_key_padding_mask=None):
                # Embed tokens
                x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
                # Scale embeddings
                x = x * math.sqrt(self.embedding.embedding_dim)
                # Apply transformer
                return self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return TransformerWithEmbedding(embedding, transformer_encoder)

    def _create_cnn_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Create a CNN encoder that embeds token ids, then applies 1D convs.

        Consumes ``(batch, seq)`` integer token ids (same input as the
        transformer encoder) and returns ``(batch, hidden_dim)``.
        """
        vocab_size = config.get("vocab_size", 257)
        embed_dim = config.get("d_model", self.hidden_dim)
        embedding = nn.Embedding(vocab_size, embed_dim)
        conv = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        class CNNWithEmbedding(nn.Module):
            def __init__(self, embedding, conv):
                super().__init__()
                self.embedding = embedding
                self.conv = conv

            def forward(self, input_ids, src_key_padding_mask=None):
                # (B, S) token ids -> (B, S, embed_dim) -> (B, embed_dim, S)
                x = self.embedding(input_ids).transpose(1, 2)
                return self.conv(x)  # (B, hidden_dim)

        return CNNWithEmbedding(embedding, conv)

    def forward(
        self,
        bytecode_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            bytecode_features: Input tensor of shape (batch_size, seq_len, feature_dim)
            attention_mask: Optional attention mask

        Returns:
            Dictionary with logits and attention weights
        """
        # Align inputs with the model's actual parameter device (callers and the
        # training loop may hand us CPU tensors even when the model is on GPU;
        # self._device can be stale if the model was moved after construction).
        from abi_reconstructor.device import to_device

        device = next(self.parameters()).device
        bytecode_features = to_device(bytecode_features, device)
        if attention_mask is not None:
            attention_mask = to_device(attention_mask, device)

        # Handle empty batch case - return empty tensors
        if bytecode_features.size(0) == 0:
            return {
                "logits": torch.zeros(0, self.num_classes, device=bytecode_features.device),
                "probs": torch.zeros(0, self.num_classes, device=bytecode_features.device),
                "confidence": torch.zeros(0, device=bytecode_features.device),
                "predicted_class": torch.zeros(0, dtype=torch.long, device=bytecode_features.device),
                "features": torch.zeros(0, self.hidden_dim, device=bytecode_features.device),
            }

        # Encode features
        if self.encoder_type == "transformer":
            # Transformer expects (batch_size, seq_len, d_model)
            if attention_mask is not None:
                # Convert to transformer mask format (True for padding)
                if attention_mask.dtype != torch.bool:
                    attention_mask_float = attention_mask.float()
                    src_key_padding_mask = attention_mask_float == 0
                else:
                    src_key_padding_mask = ~attention_mask
                encoded = self.encoder(
                    bytecode_features, src_key_padding_mask=src_key_padding_mask
                )
            else:
                encoded = self.encoder(bytecode_features)
            # Pool over sequence dimension
            if attention_mask is not None:
                # Masked mean pooling
                if attention_mask.dtype != torch.bool:
                    mask_expanded = (attention_mask != 0).float().unsqueeze(-1)
                else:
                    mask_expanded = attention_mask.float().unsqueeze(-1)
                encoded = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(
                    dim=1
                ).clamp(min=1)
            else:
                encoded = encoded.mean(dim=1)
        else:
            # CNN encoder embeds token ids internally; returns (batch, hidden_dim)
            encoded = self.encoder(bytecode_features)

        # Project encoder output to hidden_dim
        encoded = self.encoder_projection(encoded)

        # Classify
        logits = self.classifier(encoded)

        # Apply temperature scaling for calibrated logits
        calibrated_logits = logits / self.temperature.clamp(min=1e-8)

        # Compute probabilities
        probs = F.softmax(calibrated_logits, dim=-1)

        # Get confidence and predicted class
        confidence, predicted_class = torch.max(probs, dim=-1)

        return {
            "logits": logits,
            "calibrated_logits": calibrated_logits,
            "probs": probs,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "temperature": self.temperature,
            "features": encoded,
        }

    def compute_loss(
        self,
        bytecode_tokens: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute classification loss.

        Args:
            bytecode_tokens: Input bytecode tokens
            targets: Target class indices
            attention_mask: Optional attention mask
            return_outputs: Whether to return model outputs

        Returns:
            Dictionary with loss and metrics
        """
        # Forward pass (forward() aligns its inputs with the model device)
        outputs = self(bytecode_tokens, attention_mask)
        logits = outputs["logits"]

        # Apply temperature scaling
        scaled_logits = logits / self.temperature.clamp(min=1e-8)

        # Targets must live on the same device as the logits.
        from abi_reconstructor.device import to_device

        targets = to_device(targets, scaled_logits.device)

        # Compute cross-entropy loss
        loss = self.criterion(scaled_logits, targets)

        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(scaled_logits, dim=1)
            accuracy = (preds == targets).float().mean()

        result = {
            "loss": loss,
            "accuracy": accuracy,
        }

        if return_outputs:
            result["outputs"] = outputs

        return result

    def calibrate_temperature(self, val_loader, lr=0.01, max_iters=20):
        """Calibrate temperature parameter on validation set.

        Args:
            val_loader: Validation data loader
            lr: Learning rate for temperature optimization
            max_iters: Maximum iterations
        """
        # Set model to eval mode
        self.eval()

        # Create optimizer for temperature only
        optimizer = torch.optim.Adam([self.temperature], lr=lr)

        for iteration in range(max_iters):
            total_loss = 0.0
            total_samples = 0

            for batch in val_loader:
                # Move data to same device as model
                device = next(self.parameters()).device
                bytecode_tokens = batch["bytecode_tokens"].to(device)
                labels = batch["function_name_idx"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = self(bytecode_tokens, attention_mask)
                    logits = outputs["logits"]

                # Compute loss with current temperature
                scaled_logits = logits / self.temperature.clamp(min=1e-8)
                loss = self.criterion(scaled_logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clamp temperature to reasonable range
                self.temperature.data.clamp_(min=0.1, max=10.0)

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            logger.info(
                f"  Temperature calibration iteration {iteration + 1}/{max_iters}, loss: {avg_loss:.4f}, temperature: {self.temperature.item():.4f}"
            )

        # Set back to train mode
        self.train()

    def save_model(self, filepath: str):
        """Save model to file.

        Saves a complete checkpoint including architecture config so the model
        can be fully reconstructed at load time without inference.

        Args:
            filepath: Path to save model
        """
        if self.encoder_config.get("d_model") is None:
            self.encoder_config["d_model"] = self.hidden_dim

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "encoder_type": self.encoder_type,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "encoder_config": self.encoder_config,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_model(cls, filepath: str, use_cuda: bool = True):
        """Load model from file.

        Args:
            filepath: Path to load model from
            use_cuda: Whether to use CUDA

        Returns:
            Loaded FunctionNameClassifier
        """
        device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            num_classes = checkpoint.get("num_classes", 50)
            encoder_type = checkpoint.get("encoder_type", "transformer")
            hidden_dim = checkpoint.get("hidden_dim", 256)
            dropout = checkpoint.get("dropout", 0.2)
            encoder_config = checkpoint.get("encoder_config")
        else:
            state_dict = checkpoint
            num_classes = 50
            encoder_type = "transformer"
            hidden_dim = 256
            dropout = 0.2
            encoder_config = None

        if encoder_config is None:
            d_model = 64
            if "encoder.embedding.weight" in state_dict:
                d_model = state_dict["encoder.embedding.weight"].shape[1]

            hidden_dim = d_model
            if "encoder_projection.weight" in state_dict:
                hidden_dim = state_dict["encoder_projection.weight"].shape[0]

            if "classifier.3.weight" in state_dict:
                num_classes = state_dict["classifier.3.weight"].shape[0]

            layer_indices = set()
            for k in state_dict.keys():
                if "transformer.layers" in k:
                    parts = k.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            try:
                                layer_indices.add(int(parts[i + 1]))
                            except ValueError:
                                pass

            num_layers = max(layer_indices) + 1 if layer_indices else 1

            dim_feedforward = 512
            for k in state_dict.keys():
                if "linear1.weight" in k and "transformer.layers" in k:
                    dim_feedforward = state_dict[k].shape[0]
                    break

            vocab_size = 257
            if "encoder.embedding.weight" in state_dict:
                vocab_size = state_dict["encoder.embedding.weight"].shape[0]

            encoder_config = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "nhead": max(2, d_model // 64),
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
            }

        model = cls(
            num_classes=num_classes,
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_cuda=use_cuda,
            encoder_config=encoder_config,
        )

        model.load_state_dict(state_dict)
        return model

    def predict(
        self,
        bytecode_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Make predictions.

        Args:
            bytecode_features: Input features
            attention_mask: Optional attention mask
            threshold: Confidence threshold

        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(bytecode_features, attention_mask)
            logits = outputs["logits"]
            probs = F.softmax(logits / self.temperature, dim=-1)

            # Get top predictions for each sample
            top_probs, top_indices = torch.topk(probs, k=min(10, probs.size(1)), dim=1)

            predictions = []
            for i in range(bytecode_features.size(0)):
                top_probs_i = top_probs[i].cpu().numpy()
                top_indices_i = top_indices[i].cpu().numpy()

                filtered_indices = []
                filtered_probs = []
                for prob, idx in zip(top_probs_i, top_indices_i):
                    if prob >= threshold:
                        filtered_indices.append(int(idx))
                        filtered_probs.append(float(prob))

                if not filtered_probs:
                    filtered_probs = [float(top_probs_i[0])]
                    filtered_indices = [int(top_indices_i[0])]

                predictions.append({
                    "top_indices": filtered_indices,
                    "top_probs": filtered_probs,
                    "best_index": filtered_indices[0],
                    "best_prob": filtered_probs[0],
                    "confidence": filtered_probs[0],
                    "best_class_prob": filtered_probs[0],
                })

        return {
            "predictions": predictions,
            "raw_outputs": outputs,
            "confidences": torch.max(probs, dim=1)[0].cpu().numpy(),
            "mask": torch.ones(bytecode_features.size(0), dtype=torch.bool),
            "logits": logits,
            "probs": probs,
        }
