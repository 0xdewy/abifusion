"""Parameter prediction model for ABI reconstruction."""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ParameterPredictionModel(nn.Module):
    """Model for predicting function parameters from bytecode features."""

    def __init__(
        self,
        num_type_classes: int,
        max_parameters: int = 12,
        encoder_type: str = "transformer",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_function_conditioning: bool = True,
        num_function_classes: Optional[int] = None,
        use_cuda: bool = True,
        encoder_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_type_classes = num_type_classes
        self.max_parameters = max_parameters
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_function_conditioning = use_function_conditioning
        self.num_function_classes = num_function_classes
        self.use_cuda = use_cuda

        from abi_reconstructor.device import get_device_for_cuda_flag
        self._device = device or get_device_for_cuda_flag(use_cuda)

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

        # Validate required keys for transformer encoder
        if encoder_type == "transformer":
            required = {"vocab_size", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout"}
            missing = required - set(encoder_config.keys())
            if missing:
                raise ValueError(
                    f"Missing encoder_config keys for transformer: {missing}. "
                    f"Keys present: {sorted(encoder_config.keys())}"
                )

        # Create encoder based on type
        if encoder_type == "transformer":
            self.encoder = self._create_transformer_encoder(encoder_config)
        elif encoder_type == "cnn":
            self.encoder = self._create_cnn_encoder(encoder_config)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # Function conditioning embedding
        if use_function_conditioning and num_function_classes is not None:
            self.function_embedding = nn.Embedding(
                num_function_classes, hidden_dim // 4
            )
            # Projection layer for concatenated features
            self.concat_projection = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        else:
            self.function_embedding = None
            self.concat_projection = None

        # Projection from encoder output to hidden_dim
        # Encoder might output d_model (from config) which could be different from hidden_dim
        self.encoder_projection = nn.Linear(
            encoder_config.get("d_model", hidden_dim), hidden_dim
        )

        # Parameter type classifiers (one per parameter position)
        self.type_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_type_classes),
                )
                for _ in range(max_parameters)
            ]
        )

        # Parameter count predictor (classification: 0 to max_parameters)
        self.count_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_parameters + 1),  # +1 for 0 parameters
        )

        # Parameter mask predictor (binary classification for each position)
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_parameters),
        )

        # Discriminating feature projection (SigRec R11-R18 features)
        # Maps 7-feature vector to a small embedding that augments the encoder output
        self.discriminating_projection = nn.Sequential(
            nn.Linear(7, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Projects concatenated [encoder, discriminating] back to hidden_dim
        self.discriminating_concat_projection = nn.Linear(
            hidden_dim + hidden_dim // 8, hidden_dim
        )

        # Loss functions
        self.count_criterion = nn.CrossEntropyLoss()
        self.type_criterion = nn.CrossEntropyLoss()
        self.mask_criterion = nn.BCEWithLogitsLoss()

        # Class weights for type prediction (computed from dataset statistics)
        self.type_class_weights = None
        self._type_weight_tensor = None

        # Temperature parameters for calibration
        self.temperature_count = nn.Parameter(torch.tensor(1.0))
        self.temperature_type = nn.Parameter(torch.tensor(1.0))
        self.temperature_mask = nn.Parameter(torch.tensor(1.0))

        # Route device selection through DeviceManager
        if self._device.type != "cpu":
            self.to(self._device)

    def _create_transformer_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Create transformer encoder."""
        # Create embedding layer
        vocab_size = config.get("vocab_size", 257)
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
        """Create CNN encoder."""
        return nn.Sequential(
            nn.Conv1d(config.get("input_channels", 256), 128, kernel_size=3, padding=1),
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

    def forward(
        self,
        bytecode_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        function_ids: Optional[torch.Tensor] = None,
        discriminating_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            bytecode_features: Input tensor of shape (batch_size, seq_len, feature_dim)
            attention_mask: Optional attention mask
            function_ids: Optional function IDs for conditioning
            discriminating_features: Optional SigRec R11-R18 feature vector
                of shape (batch_size, 7). Inferred from bytecode via
                DiscriminatingFeatureExtractor.

        Returns:
            Dictionary with type logits, mask logits, count predictions.
        """
        # Handle empty batch case - return empty tensors
        if bytecode_features.size(0) == 0:
            return {
                "count_logits": torch.zeros(0, self.max_parameters + 1, device=bytecode_features.device),
                "count_probs": torch.zeros(0, self.max_parameters + 1, device=bytecode_features.device),
                "count_confidence": torch.zeros(0, device=bytecode_features.device),
                "predicted_count": torch.zeros(0, dtype=torch.long, device=bytecode_features.device),
                "type_logits": torch.zeros(0, self.max_parameters, self.num_type_classes, device=bytecode_features.device),
                "type_probs": torch.zeros(0, self.max_parameters, self.num_type_classes, device=bytecode_features.device),
                "type_confidences": torch.zeros(0, self.max_parameters, device=bytecode_features.device),
                "predicted_types": torch.zeros(0, self.max_parameters, dtype=torch.long, device=bytecode_features.device),
                "mask_logits": torch.zeros(0, self.max_parameters, device=bytecode_features.device),
                "mask_probs": torch.zeros(0, self.max_parameters, device=bytecode_features.device),
                "predicted_mask": torch.zeros(0, self.max_parameters, dtype=torch.long, device=bytecode_features.device),
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
            # CNN expects (batch_size, channels, seq_len)
            encoded = bytecode_features.transpose(1, 2)
            encoded = self.encoder(encoded)

        # Project encoder output to hidden_dim
        encoded = self.encoder_projection(encoded)

        # Apply function conditioning if enabled
        if (
            self.use_function_conditioning
            and function_ids is not None
            and self.function_embedding is not None
            and self.concat_projection is not None
        ):
            # Handle function_ids shape - must be 1D (batch,)
            # If 2D, take only the first element of each row to get 1D
            if function_ids.dim() > 1:
                function_ids = function_ids[:, 0]
            function_emb = self.function_embedding(function_ids)
            # Concatenate function embedding with encoded features
            encoded = torch.cat([encoded, function_emb], dim=1)
            # Project back to hidden_dim using the pre-defined layer
            encoded = self.concat_projection(encoded)

        # Apply discriminating features (SigRec R11-R18) if provided
        if discriminating_features is not None:
            disc_emb = self.discriminating_projection(discriminating_features)
            encoded = torch.cat([encoded, disc_emb], dim=1)
            encoded = self.discriminating_concat_projection(encoded)

        # Get type logits for each parameter position
        type_logits = []
        for classifier in self.type_classifiers:
            type_logits.append(classifier(encoded))
        type_logits = torch.stack(
            type_logits, dim=1
        )  # (batch_size, max_parameters, num_type_classes)

        # Get count logits
        count_logits = self.count_predictor(encoded)  # (batch_size, max_parameters + 1)

        # Get mask logits
        mask_logits = self.mask_predictor(encoded)  # (batch_size, max_parameters)

        # Compute probabilities and predictions
        count_probs = F.softmax(count_logits, dim=-1)
        count_confidence, predicted_count = torch.max(count_probs, dim=-1)

        type_probs_list = []
        type_confidences_list = []
        predicted_types_list = []
        for i in range(len(self.type_classifiers)):
            type_prob = F.softmax(type_logits[:, i, :], dim=-1)
            type_conf, type_pred = torch.max(type_prob, dim=-1)
            type_probs_list.append(type_prob)
            type_confidences_list.append(type_conf)
            predicted_types_list.append(type_pred)

        type_probs = torch.stack(type_probs_list, dim=1)
        type_confidences = torch.stack(type_confidences_list, dim=1)
        predicted_types = torch.stack(predicted_types_list, dim=1)

        mask_probs = torch.sigmoid(mask_logits)
        predicted_mask = (mask_probs > 0.5).long()

        return {
            "count_logits": count_logits,
            "count_probs": count_probs,
            "count_confidence": count_confidence,
            "predicted_count": predicted_count,
            "type_logits": type_logits,
            "type_probs": type_probs,
            "type_confidences": type_confidences,
            "predicted_types": predicted_types,
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
            "predicted_mask": predicted_mask,
            "features": encoded,
        }

    def compute_loss(
        self,
        bytecode_tokens: torch.Tensor,
        parameter_count: torch.Tensor,
        parameter_types: torch.Tensor,
        parameter_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        function_name_idx: Optional[torch.Tensor] = None,
        discriminating_features: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        return_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute parameter prediction loss.

        Args:
            bytecode_tokens: Input bytecode tokens
            parameter_count: Target parameter counts
            parameter_types: Target parameter types
            parameter_mask: Target parameter mask (1 for parameter exists, 0 for padding)
            attention_mask: Optional attention mask
            function_name_idx: Optional function name indices for conditioning
            discriminating_features: Optional SigRec R11-R18 feature vector (batch_size, 7)
            loss_weights: Optional dictionary with weights for count, type, mask losses
            return_outputs: Whether to return model outputs

        Returns:
            Dictionary with losses and metrics
        """
        # Forward pass
        outputs = self(bytecode_tokens, attention_mask, function_name_idx, discriminating_features)

        # Get logits from outputs
        count_logits = outputs["count_logits"]
        type_logits = outputs["type_logits"]
        mask_logits = outputs["mask_logits"]

        # Truncate parameter_types and parameter_mask to max_parameters
        # The dataset might have more parameters than our model can handle
        parameter_types = parameter_types[:, : self.max_parameters]
        parameter_mask = parameter_mask[:, : self.max_parameters]

        # Default loss weights
        if loss_weights is None:
            loss_weights = {"count": 1.0, "type": 1.0, "mask": 0.5}

        # Count prediction loss (classification)
        # Clip parameter_count to valid range [0, max_parameters]
        parameter_count_clipped = torch.clamp(parameter_count, max=self.max_parameters)

        count_loss = self.count_criterion(count_logits, parameter_count_clipped)
        with torch.no_grad():
            count_preds = torch.argmax(count_logits, dim=1)
            count_accuracy = (count_preds == parameter_count_clipped).float().mean()

        # Type classification loss (only for positions where parameter exists)
        type_loss = 0.0
        type_correct = 0
        type_total = 0
        num_valid_positions = 0

        for i in range(self.max_parameters):
            pos_type_logits = type_logits[:, i, :]
            pos_targets = parameter_types[:, i]
            pos_mask = parameter_mask[:, i]

            if pos_mask.sum() > 0:
                valid_idx = pos_mask.nonzero(as_tuple=True)[0]
                if len(valid_idx) > 0:
                    valid_logits = pos_type_logits[valid_idx]
                    valid_targets = pos_targets[valid_idx]

                    # Use weighted criterion if available
                    if self.type_class_weights is not None and hasattr(self, 'type_criterion_weighted'):
                        pos_loss = self.type_criterion_weighted(
                            valid_logits, valid_targets
                        )
                    else:
                        pos_loss = self.type_criterion(valid_logits, valid_targets)
                    type_loss += pos_loss
                    num_valid_positions += 1

                    with torch.no_grad():
                        pos_preds = torch.argmax(valid_logits, dim=1)
                        pos_correct = (pos_preds == valid_targets).sum().item()
                        type_correct += pos_correct
                        type_total += len(valid_idx)

        if num_valid_positions > 0:
            type_loss = type_loss / num_valid_positions
            type_accuracy = torch.tensor(
                type_correct / type_total, device=type_logits.device
            )
        else:
            type_loss = torch.tensor(0.0, device=type_logits.device)
            type_accuracy = torch.tensor(0.0, device=type_logits.device)

        # Mask prediction loss (binary classification)
        mask_loss = self.mask_criterion(mask_logits, parameter_mask)
        with torch.no_grad():
            mask_preds = (torch.sigmoid(mask_logits) > 0.5).float()
            mask_accuracy = (mask_preds == parameter_mask).float().mean()

        # Total loss (weighted sum)
        total_loss = (
            loss_weights["count"] * count_loss
            + loss_weights["type"] * type_loss
            + loss_weights["mask"] * mask_loss
        )

        result = {
            "total_loss": total_loss,
            "count_loss": count_loss,
            "type_loss": type_loss,
            "mask_loss": mask_loss,
            "count_accuracy": count_accuracy,
            "type_accuracy": type_accuracy,
            "mask_accuracy": mask_accuracy,
            "loss_weights": loss_weights,
        }

        if return_outputs:
            result["outputs"] = outputs

        return result

    def predict(
        self,
        bytecode_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        count_threshold: float = 0.5,
        type_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Make parameter predictions.

        Args:
            bytecode_features: Input features
            attention_mask: Optional attention mask
            count_threshold: Confidence threshold for count prediction
            type_threshold: Confidence threshold for type prediction
            mask_threshold: Threshold for parameter existence

        Returns:
            Dictionary with predictions in nested per-sample format
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(bytecode_features, attention_mask)

            count_probs = outputs["count_probs"]
            count_confidence = outputs["count_confidence"]
            predicted_count = outputs["predicted_count"]
            type_probs = outputs["type_probs"]
            type_confidences = outputs["type_confidences"]
            predicted_types = outputs["predicted_types"]
            mask_probs = outputs["mask_probs"]

        batch_size = bytecode_features.size(0)
        predictions_list = []

        for i in range(batch_size):
            valid_counts = []
            valid_count_probs = []
            for c in range(self.max_parameters + 1):
                prob = count_probs[i, c].item()
                if prob >= count_threshold:
                    valid_counts.append(c)
                    valid_count_probs.append(prob)

            best_count = predicted_count[i].item()
            best_count_prob = count_probs[i, best_count].item()

            type_preds_for_sample = []
            for pos in range(self.max_parameters):
                type_prob = type_probs[i, pos]
                mask_prob = mask_probs[i, pos].item()
                is_valid = mask_prob >= mask_threshold

                valid_types = []
                valid_type_probs = []
                for t in range(self.num_type_classes):
                    prob = type_prob[t].item()
                    if prob >= type_threshold:
                        valid_types.append(t)
                        valid_type_probs.append(prob)

                best_type = predicted_types[i, pos].item()
                best_type_prob = type_prob[best_type].item()

                type_preds_for_sample.append({
                    "position": pos,
                    "valid_types": valid_types,
                    "valid_type_probs": valid_type_probs,
                    "best_type": best_type,
                    "best_type_prob": best_type_prob,
                    "mask_prob": mask_prob,
                    "is_valid": is_valid,
                })

            avg_type_conf = sum(t["best_type_prob"] for t in type_preds_for_sample) / len(type_preds_for_sample) if type_preds_for_sample else 0.0

            predictions_list.append({
                "valid_counts": valid_counts,
                "valid_count_probs": valid_count_probs,
                "best_count": best_count,
                "best_count_prob": best_count_prob,
                "type_predictions": type_preds_for_sample,
                "count_confidence": count_confidence[i].item(),
                "avg_type_confidence": avg_type_conf,
            })

        return {
            "predictions": predictions_list,
            "raw_outputs": {
                "count_logits": outputs["count_logits"],
                "count_probs": count_probs,
                "count_confidence": count_confidence,
                "predicted_count": predicted_count,
                "type_logits": outputs["type_logits"],
                "type_probs": type_probs,
                "type_confidences": type_confidences,
                "predicted_types": predicted_types,
                "mask_logits": outputs["mask_logits"],
                "mask_probs": mask_probs,
                "features": outputs["features"],
            },
        }

    def calibrate_temperatures(self, val_loader, lr=0.01, max_iters=20):
        """Calibrate temperature parameters on validation set.

        Args:
            val_loader: Validation data loader
            lr: Learning rate for temperature optimization
            max_iters: Maximum iterations
        """
        self.eval()

        optimizer = torch.optim.Adam(
            [self.temperature_count, self.temperature_type, self.temperature_mask],
            lr=lr
        )

        for _ in range(max_iters):
            for batch in val_loader:
                bytecode_tokens = batch["bytecode_tokens"]
                parameter_count = batch["parameter_count"]
                attention_mask = batch.get("attention_mask")

                outputs = self(bytecode_tokens, attention_mask)
                count_logits = outputs["count_logits"]

                clipped_count = torch.clamp(parameter_count, max=self.max_parameters)
                loss = self.count_criterion(count_logits, clipped_count)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for temp in [self.temperature_count, self.temperature_type, self.temperature_mask]:
                    temp.data.clamp_(min=0.1, max=10.0)

        self.train()

    def set_type_class_weights(self, class_weights: torch.Tensor):
        """Set class weights for type prediction loss.

        Args:
            class_weights: Tensor of shape (num_type_classes,) with weights for each class
        """
        self.type_class_weights = class_weights
        weight_device = (
            next(self.count_criterion.parameters()).device
            if hasattr(self.count_criterion, "weight") and self.count_criterion.weight is not None
            else torch.device("cpu")
        )
        self._type_weight_tensor = class_weights.to(weight_device)
        # Create weighted loss criterion
        self.type_criterion_weighted = nn.CrossEntropyLoss(
            weight=self._type_weight_tensor
        )

    def save_model(self, path):
        """Save model to file.

        Saves a complete checkpoint including architecture config so the model
        can be fully reconstructed at load time without inference.

        Args:
            path: Path to save model
        """
        self.encoder_config.setdefault("d_model", self.hidden_dim)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_type_classes": self.num_type_classes,
            "max_parameters": self.max_parameters,
            "encoder_type": self.encoder_type,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "use_function_conditioning": self.use_function_conditioning,
            "num_function_classes": self.num_function_classes,
            "encoder_config": self.encoder_config,
            "format_version": 2,
            "device": str(self._device),
        }
        torch.save(checkpoint, path)
        logger.info(f"  Model saved to {path}")

    @classmethod
    def load_from_checkpoint(
        cls,
        filepath: str,
        device: Optional[torch.device] = None,
    ):
        """Load model from a self-contained checkpoint saved by save_model().

        Reads all architecture from the checkpoint — no caller-supplied guesses.

        Args:
            filepath: Path to checkpoint saved by save_model().
            device: Device to load to (default: cpu).

        Returns:
            Fully reconstructed ParameterPredictionModel.
        """
        checkpoint = torch.load(filepath, map_location=device or "cpu", weights_only=False)

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dict saved by save_model()")

        use_cuda = str(checkpoint.get("device", device or "cpu")) != "cpu"

        model = cls(
            num_type_classes=checkpoint["num_type_classes"],
            max_parameters=checkpoint["max_parameters"],
            encoder_type=checkpoint.get("encoder_type", "transformer"),
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint.get("dropout", 0.2),
            use_function_conditioning=checkpoint.get("use_function_conditioning", True),
            num_function_classes=checkpoint.get("num_function_classes"),
            encoder_config=checkpoint["encoder_config"],
            use_cuda=use_cuda,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    @classmethod
    def load_model(
        cls,
        filepath: str,
        num_type_classes: int = 24,
        max_parameters: int = 12,
        encoder_type: str = "transformer",
        hidden_dim: int = 128,
        use_function_conditioning: bool = True,
        num_function_classes: int = 47,
        use_cuda: bool = True,
        encoder_config: Optional[Dict[str, Any]] = None,
    ):
        """Load model from file.

        Args:
            filepath: Path to model checkpoint
            num_type_classes: Number of parameter type classes
            max_parameters: Maximum number of parameters
            encoder_type: Type of encoder (transformer or cnn)
            hidden_dim: Hidden dimension size
            use_function_conditioning: Whether to use function conditioning
            num_function_classes: Number of function classes for conditioning
            use_cuda: Whether to use CUDA
            encoder_config: Optional encoder configuration

        Returns:
            Loaded ParameterPredictionModel
        """
        # Load checkpoint
        device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        # Handle both dict format (with model_state_dict key) and plain state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "encoder_config" in checkpoint:
                encoder_config = checkpoint["encoder_config"]
            if "hidden_dim" in checkpoint:
                hidden_dim = checkpoint["hidden_dim"]
            if "num_type_classes" in checkpoint:
                num_type_classes = checkpoint["num_type_classes"]
            if "max_parameters" in checkpoint:
                max_parameters = checkpoint["max_parameters"]
            if "use_function_conditioning" in checkpoint:
                use_function_conditioning = checkpoint["use_function_conditioning"]
            if "num_function_classes" in checkpoint:
                num_function_classes = checkpoint["num_function_classes"]
        else:
            state_dict = checkpoint

        # Infer num_type_classes from type_classifiers
        for k in state_dict.keys():
            if k.startswith("type_classifiers.") and k.endswith(".3.weight"):
                inferred_num_types = state_dict[k].shape[0]
                if inferred_num_types != num_type_classes:
                    logger.info(f"  Note: Overriding num_type_classes from {num_type_classes} to {inferred_num_types}")
                    num_type_classes = inferred_num_types

        # Infer max_parameters from number of type_classifiers
        max_params = 0
        for k in state_dict.keys():
            if k.startswith("type_classifiers.") and ".0." not in k:
                parts = k.split(".")
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        max_params = max(max_params, idx + 1)
                    except ValueError:
                        pass
        if max_params > 0 and max_params != max_parameters:
            logger.info(f"  Note: Overriding max_parameters from {max_parameters} to {max_params}")
            max_parameters = max_params

        # Infer vocab_size from encoder.embedding.weight
        if "encoder.embedding.weight" in state_dict:
            inferred_vocab_size = state_dict["encoder.embedding.weight"].shape[0]
            if encoder_config is None:
                encoder_config = {}
            if "vocab_size" not in encoder_config or encoder_config.get("vocab_size") != inferred_vocab_size:
                logger.info(f"  Note: Setting vocab_size to {inferred_vocab_size} based on checkpoint")
                encoder_config["vocab_size"] = inferred_vocab_size
            elif encoder_config.get("vocab_size") != inferred_vocab_size:
                logger.info(f"  Note: Overriding vocab_size from {encoder_config.get('vocab_size')} to {inferred_vocab_size}")
                encoder_config["vocab_size"] = inferred_vocab_size

        # Infer num_function_classes from function_embedding.weight
        if "function_embedding.weight" in state_dict:
            inferred_num_func_classes = state_dict["function_embedding.weight"].shape[0]
            if inferred_num_func_classes != num_function_classes:
                logger.info(f"  Note: Overriding num_function_classes from {num_function_classes} to {inferred_num_func_classes}")
                num_function_classes = inferred_num_func_classes

        # Infer hidden_dim from concat_projection (more reliable than embedding)
        if "concat_projection.weight" in state_dict:
            # concat_projection.weight shape: [hidden_dim, hidden_dim + func_dim]
            inferred_hidden_dim = state_dict["concat_projection.weight"].shape[0]
            if inferred_hidden_dim != hidden_dim:
                logger.info(
                    f"  Note: Overriding hidden_dim from {hidden_dim} to {inferred_hidden_dim} based on concat_projection"
                )
                hidden_dim = inferred_hidden_dim

            # Infer number of transformer layers and d_model
            if encoder_type == "transformer":
                layer_indices = set()
                d_model = None
                dim_feedforward = None
                for k in state_dict.keys():
                    if "transformer.layers" in k:
                        parts = k.split(".")
                        for i, part in enumerate(parts):
                            if (
                                part == "layers"
                                and i + 1 < len(parts)
                                and parts[i + 1].isdigit()
                            ):
                                layer_indices.add(int(parts[i + 1]))
                    # Try to infer d_model from embedding or transformer weights
                    if "encoder.embedding.weight" in state_dict:
                        d_model = state_dict["encoder.embedding.weight"].shape[1]
                    elif (
                        "encoder.transformer.layers.0.self_attn.out_proj.weight"
                        in state_dict
                    ):
                        d_model = state_dict[
                            "encoder.transformer.layers.0.self_attn.out_proj.weight"
                        ].shape[0]
                    # Infer dim_feedforward from linear1 bias
                    if "encoder.transformer.layers.0.linear1.bias" in state_dict:
                        dim_feedforward = state_dict["encoder.transformer.layers.0.linear1.bias"].shape[0]

                num_layers = max(layer_indices) + 1 if layer_indices else 1
                if d_model is None:
                    d_model = hidden_dim  # Fallback

                if encoder_config is None:
                    encoder_config = {
                        "vocab_size": 257,
                        "d_model": d_model,  # Use inferred d_model, not necessarily hidden_dim
                        "nhead": max(2, d_model // 64),
                        "num_layers": num_layers,
                        "dim_feedforward": d_model * 4,
                        "dropout": 0.2,
                    }
                else:
                    # Update encoder_config with inferred values
                    if (
                        "d_model" not in encoder_config
                        or encoder_config["d_model"] != d_model
                    ):
                        logger.info(
                            f"  Note: Setting d_model to {d_model} based on checkpoint"
                        )
                        encoder_config["d_model"] = d_model
                    if (
                        "num_layers" not in encoder_config
                        or encoder_config["num_layers"] != num_layers
                    ):
                        logger.info(
                            f"  Note: Setting num_layers to {num_layers} based on checkpoint"
                        )
                        encoder_config["num_layers"] = num_layers
                if dim_feedforward is not None:
                    encoder_config["dim_feedforward"] = dim_feedforward
                    logger.info(f"  Note: Setting dim_feedforward to {dim_feedforward} based on checkpoint")
                elif "dim_feedforward" not in encoder_config:
                    encoder_config["dim_feedforward"] = d_model * 4
                if "dropout" not in encoder_config:
                    encoder_config["dropout"] = 0.2
                if "nhead" not in encoder_config:
                    encoder_config["nhead"] = max(2, d_model // 64)

        # Create model instance with inferred parameters
        device_obj = torch.device(device) if device else torch.device("cpu")
        model = cls(
            num_type_classes=num_type_classes,
            max_parameters=max_parameters,
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            use_function_conditioning=use_function_conditioning,
            num_function_classes=num_function_classes,
            use_cuda=use_cuda,
            encoder_config=encoder_config,
            device=device_obj,
        )

        # Load state dict
        model.load_state_dict(state_dict)

        if device_obj.type != "cpu":
            model.to(device_obj)

        model.eval()
        return model
