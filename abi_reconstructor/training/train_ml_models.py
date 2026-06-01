"""Training script for the pure ML two-model ABI reconstruction system.

Trains both models:
1. FunctionNameClassifier (Model 1)
2. ParameterPredictionModel (Model 2)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from abi_reconstructor.device import batch_to_device
from abi_reconstructor.models.function_name_classifier import FunctionNameClassifier
from abi_reconstructor.models.parameter_prediction_model import ParameterPredictionModel
from abi_reconstructor.training.parameter_dataset import (
    ParameterDataset,
    ParameterDatasetTorch,
)

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when a timeout occurs."""

    pass


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB.

    Returns:
        Dictionary with RAM and GPU memory usage
    """
    memory_info = {}

    # Get RAM usage
    process = psutil.Process(os.getpid())
    memory_info["ram_mb"] = process.memory_info().rss / 1024 / 1024

    # Get GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info[f"gpu_{i}_allocated_mb"] = (
                torch.cuda.memory_allocated(i) / 1024 / 1024
            )
            memory_info[f"gpu_{i}_cached_mb"] = (
                torch.cuda.memory_reserved(i) / 1024 / 1024
            )

    return memory_info


def log_memory_usage(prefix: str = ""):
    """Log current memory usage.

    Args:
        prefix: Prefix for log message
    """
    memory = get_memory_usage()
    message = f"{prefix}Memory usage: "
    message += f"RAM: {memory['ram_mb']:.1f}MB"

    if "gpu_0_allocated_mb" in memory:
        message += f", GPU0: {memory['gpu_0_allocated_mb']:.1f}MB allocated, {memory['gpu_0_cached_mb']:.1f}MB cached"

    logger.info(message)


class ModelTrainer:
    """Base trainer for ML models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        use_cuda: bool = True,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
    ):
        """Initialize model trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            use_cuda: Whether to use GPU
            checkpoint_dir: Directory for checkpoints
            model_name: Name for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device through DeviceManager
        from abi_reconstructor.device import get_device_manager
        self._dm = get_device_manager()
        if self._dm.is_cuda:
            self.model.to(self._dm.device)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Log memory at start of epoch
        log_memory_usage("Start of epoch: ")

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()

            # Simple timeout check without signal
            batch_timeout = 120  # 2 minutes max per batch
            timeout_start = time.time()

            # Move batch to device
            batch = batch_to_device(batch)

            # Check timeout before forward pass
            if time.time() - timeout_start > batch_timeout:
                logger.info(
                    f"\n❌ ERROR: Batch {batch_idx + 1} timed out during data loading!"
                )
                raise RuntimeError(
                    f"Batch processing timed out after {batch_timeout} seconds"
                )

            # Call compute_batch_loss with timeout wrapper
            def compute_loss_with_timeout(current_batch):
                return self.compute_batch_loss(current_batch)

            # Simple timeout by checking elapsed time
            try:
                loss_dict = compute_loss_with_timeout(batch)
            except Exception:
                import traceback

                traceback.print_exc()
                raise

            # Check timeout after forward pass
            if time.time() - timeout_start > batch_timeout:
                logger.info(
                    f"\n❌ ERROR: Batch {batch_idx + 1} timed out during forward pass!"
                )
                logger.info("   The parameter model forward pass is taking too long.")
                logger.info("   This could be due to:")
                logger.info("   1. Very large model architecture")
                logger.info("   2. Complex computations in forward()")
                logger.info("   3. Data shape issues")
                raise RuntimeError(
                    f"Forward pass timed out after {batch_timeout} seconds"
                )

            loss = loss_dict["total_loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Check timeout after backward pass
            if time.time() - timeout_start > batch_timeout:
                logger.info(
                    f"\n❌ ERROR: Batch {batch_idx + 1} timed out during backward pass!"
                )
                raise RuntimeError(
                    f"Backward pass timed out after {batch_timeout} seconds"
                )

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            n_batches += 1

            batch_time = time.time() - batch_start_time
            if batch_idx == 0:
                # Estimate total epoch time
                estimated_epoch_time = batch_time * len(self.train_loader)
                logger.info(
                    f"  Estimated epoch time: ~{estimated_epoch_time / 60:.1f} minutes"
                )

            # Log progress (more frequently for debugging)
            if (batch_idx + 1) % 2 == 0 or batch_idx < 3:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, Time: {batch_time:.1f}s"
                )
                sys.stdout.flush()
                # Log memory every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    log_memory_usage(f"  Batch {batch_idx + 1}: ")

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        # Log memory at end of epoch
        log_memory_usage("End of epoch: ")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss

    def compute_batch_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model on validation set.

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        metrics = {}

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch_to_device(batch)

                # Forward pass
                loss_dict = self.compute_batch_loss(batch)
                loss = loss_dict["total_loss"]

                # Update statistics
                total_loss += loss.item()
                n_batches += 1

                # Update metrics
                for key, value in loss_dict.items():
                    if key != "total_loss" and isinstance(value, torch.Tensor):
                        metrics[key] = metrics.get(key, 0.0) + value.item()

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        # Average metrics
        for key in metrics:
            metrics[key] = metrics[key] / n_batches

        return avg_loss, metrics

    def train(
        self,
        num_epochs: int = 10,
        patience: int = 10,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            patience: Early stopping patience
            save_best: Whether to save best model

        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.model_name}...")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Patience: {patience}")
        logger.info(f"  Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(self.val_loader.dataset)}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time

            # Validate
            val_loss, val_metrics = self.validate()

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)
            self.history["learning_rates"].append(current_lr)

            # Print progress
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            logger.info(f"  Time: {train_time:.2f}s")

            if val_metrics:
                logger.info("  Validation Metrics:")
                for key, value in val_metrics.items():
                    logger.info(f"    {key}: {value:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if save_best:
                    self.save_checkpoint(f"best_{self.model_name}.pth")
                    logger.info(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"  No improvement for {patience_counter}/{patience} epochs")

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Save final model
        self.save_checkpoint(f"final_{self.model_name}.pth")

        # Save training history
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        # Convert tensors to lists for JSON serialization
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                serializable_history[key] = [
                    v.item() if isinstance(v, torch.Tensor) else v for v in value
                ]
            else:
                serializable_history[key] = value

        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

        logger.info(f"\nTraining completed for {self.model_name}")
        logger.info(f"  Best validation loss: {best_val_loss:.4f}")
        logger.info(f"  Checkpoints saved to: {self.checkpoint_dir}")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint with full architecture for reconstruction."""
        checkpoint_path = self.checkpoint_dir / filename
        model = self.model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "history": self.history,
            # Architecture — needed to reconstruct model without guessing
            "num_type_classes": getattr(model, "num_type_classes", None),
            "max_parameters": getattr(model, "max_parameters", None),
            "encoder_type": getattr(model, "encoder_type", "transformer"),
            "hidden_dim": getattr(model, "hidden_dim", None),
            "dropout": getattr(model, "dropout", 0.2),
            "use_function_conditioning": getattr(model, "use_function_conditioning", True),
            "num_function_classes": getattr(model, "num_function_classes", None),
            "encoder_config": getattr(model, "encoder_config", {}),
            "format_version": 2,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(
            checkpoint_path, map_location="cuda" if self.use_cuda else "cpu"
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class FunctionNameTrainer(ModelTrainer):
    """Trainer for FunctionNameClassifier."""

    def compute_batch_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute classification loss for function name prediction."""
        bytecode_tokens = batch["bytecode_tokens"]
        labels = batch["function_name_idx"]
        attention_mask = batch.get("attention_mask")

        loss_dict = self.model.compute_loss(
            bytecode_tokens, labels, attention_mask, return_outputs=False
        )

        # Rename 'loss' to 'total_loss' for consistency with base trainer
        if "loss" in loss_dict:
            loss_dict["total_loss"] = loss_dict.pop("loss")

        return loss_dict


class ParameterPredictionTrainer(ModelTrainer):
    """Trainer for ParameterPredictionModel."""

    def compute_batch_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss for parameter prediction."""
        bytecode_tokens = batch["bytecode_tokens"]
        parameter_count = batch["parameter_count"]
        parameter_types = batch["parameter_types"]
        parameter_mask = batch["parameter_mask"]
        attention_mask = batch.get("attention_mask")

        # Get function indices if available
        function_name_idx = batch.get("function_name_idx")
        # Get discriminating features (SigRec R11-R18) if available
        discriminating_features = batch.get("discriminating_features")

        loss_dict = self.model.compute_loss(
            bytecode_tokens,
            parameter_count,
            parameter_types,
            parameter_mask,
            attention_mask,
            function_name_idx,
            discriminating_features=discriminating_features,
            return_outputs=False,
        )

        return loss_dict


def prepare_datasets(
    data_dir: str = "data",
    max_samples: int = 0,  # 0 = unlimited, use all available samples
    batch_size: int = 32,
    test_split: float = 0.2,
    val_split: float = 0.1,
    seed: int = 42,
    disable_cache: bool = False,
    filter_noisy: bool = True,
    min_params: int = 0,
    max_params: int = 12,
) -> Tuple[DataLoader, DataLoader, DataLoader, ParameterDataset, List[Dict[str, Any]]]:
    """Prepare datasets for training.

    Args:
        data_dir: Data directory
        max_samples: Maximum number of samples (0 = unlimited, use all available)
        batch_size: Batch size
        test_split: Fraction for test set
        val_split: Fraction of training for validation
        seed: Random seed
        filter_noisy: Whether to filter noisy samples
        min_params: Minimum number of parameters to keep sample
        max_params: Maximum number of parameters to keep sample

    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset, train_samples)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    logger.info("Loading parameter dataset...")
    dataset = ParameterDataset(data_dir=data_dir)
    samples = dataset.load_parameter_samples(max_samples=max_samples)

    logger.info(f"Loaded {len(samples)} samples")

    # Filter noisy samples before splitting
    if filter_noisy:
        logger.info("Filtering noisy samples...")
        samples = dataset.filter_noisy_samples(
            samples, min_params=min_params, max_params=max_params
        )
        logger.info(f"After filtering: {len(samples)} samples")

    # Prepare training data splits
    train_data, val_data, test_data = dataset.prepare_training_data(
        samples, test_split=test_split, val_split=val_split
    )

    # Extract train samples for class weight computation
    train_samples = train_data.get("samples", [])

    # Create PyTorch datasets with optional caching
    use_cache = not disable_cache
    train_dataset = ParameterDatasetTorch(
        train_data,
        tokenize_bytecode=True,
        cache_dir="cache/tokenized_bytecode",
        use_cache=use_cache,
    )
    val_dataset = ParameterDatasetTorch(
        val_data,
        tokenize_bytecode=True,
        cache_dir="cache/tokenized_bytecode",
        use_cache=use_cache,
    )
    test_dataset = ParameterDatasetTorch(
        test_data,
        tokenize_bytecode=True,
        cache_dir="cache/tokenized_bytecode",
        use_cache=use_cache,
    )

    # Create data loaders - use 0 workers to avoid file descriptor issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ParameterDatasetTorch.collate_fn,
        num_workers=0,  # Changed from 2 to avoid "Too many open files"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ParameterDatasetTorch.collate_fn,
        num_workers=0,  # Changed from 2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ParameterDatasetTorch.collate_fn,
        num_workers=0,  # Changed from 2
    )

    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, dataset, train_samples


def train_function_name_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset: ParameterDataset,
    use_cuda: bool = True,
    checkpoint_dir: str = "checkpoints",
    num_epochs: int = 10,
    encoder_type: str = "transformer",
    encoder_config: Optional[Dict[str, Any]] = None,
    hidden_dim: int = -1,
) -> FunctionNameClassifier:
    """Train function name classifier (Model 1).

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        dataset: ParameterDataset for vocabulary
        use_cuda: Whether to use GPU
        checkpoint_dir: Directory for checkpoints

    Returns:
        Trained FunctionNameClassifier
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training FunctionNameClassifier (Model 1)")
    logger.info("=" * 60)

    # Log initial memory usage
    log_memory_usage("Before loading function classifier: ")

    # Model parameters
    num_classes = len(dataset.FUNCTION_NAME_VOCAB) + 1  # +1 for unknown

    # Set default encoder config if not provided
    if encoder_config is None:
        if encoder_type == "transformer":
            encoder_config = {
                "vocab_size": 257,
                "d_model": 128,  # Reduced from 256
                "nhead": 4,  # Reduced from 8
                "num_layers": 2,  # Reduced from 4
                "dim_feedforward": 512,  # Reduced from 1024
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": use_cuda,
                "use_gradient_checkpointing": False,  # Can be enabled for memory saving
            }
        else:  # cnn
            encoder_config = {
                "vocab_size": 257,
                "embedding_dim": 128,
                "num_filters": 128,
                "filter_sizes": (3, 4, 5),
                "dropout": 0.1,
                "use_cuda": use_cuda,
            }

    # Set hidden_dim if not provided
    if hidden_dim <= 0:
        hidden_dim = 128  # Default

    logger.info("Creating FunctionNameClassifier with:")
    logger.info(f"  Encoder type: {encoder_type}")
    logger.info(f"  Hidden dimension: {hidden_dim}")
    logger.info(f"  Number of classes: {num_classes}")
    if encoder_config and encoder_type == "transformer":
        logger.info(
            f"  Transformer config: d_model={encoder_config.get('d_model', 'N/A')}, "
            f"nhead={encoder_config.get('nhead', 'N/A')}, "
            f"num_layers={encoder_config.get('num_layers', 'N/A')}"
        )

    # Create model
    model = FunctionNameClassifier(
        num_classes=num_classes,
        encoder_type=encoder_type,
        encoder_config=encoder_config,
        hidden_dim=hidden_dim,
        dropout=0.2,
        use_cuda=use_cuda,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Create trainer
    trainer = FunctionNameTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        use_cuda=use_cuda,
        checkpoint_dir=checkpoint_dir,
        model_name="function_classifier",
    )

    # Train model
    trainer.train(
        num_epochs=num_epochs,
        patience=min(10, num_epochs // 2),
        save_best=True,
    )

    # Calibrate temperature on validation set (if available)
    if len(val_loader) > 0:
        logger.info("\nCalibrating temperature parameter...")
        model.calibrate_temperature(val_loader, lr=0.01, max_iters=20)
    else:
        logger.info("\nSkipping temperature calibration (no validation data)")

    # Save final model
    model_path = Path(checkpoint_dir) / "function_classifier_final.pth"
    model.save_model(str(model_path))

    return model


def train_parameter_prediction_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset: ParameterDataset,
    use_cuda: bool = True,
    checkpoint_dir: str = "checkpoints",
    num_epochs: int = 10,
    encoder_type: str = "transformer",
    encoder_config: Optional[Dict[str, Any]] = None,
    hidden_dim: int = -1,
    max_parameters: int = -1,
    train_samples: Optional[List[Dict[str, Any]]] = None,
    use_class_weights: bool = True,
) -> ParameterPredictionModel:
    """Train parameter prediction model (Model 2).

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        dataset: ParameterDataset for type vocabulary
        use_cuda: Whether to use GPU
        checkpoint_dir: Directory for checkpoints
        train_samples: Original training samples for computing class weights
        use_class_weights: Whether to use class weighting for type prediction

    Returns:
        Trained ParameterPredictionModel
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training ParameterPredictionModel (Model 2)")
    logger.info("=" * 60)

    # Log initial memory usage
    log_memory_usage("Before loading parameter model: ")

    # Model parameters
    num_type_classes = len(dataset.COMMON_TYPES) + 1  # +1 for unknown
    num_function_classes = len(dataset.FUNCTION_NAME_VOCAB) + 1

    # Set default encoder config if not provided
    if encoder_config is None:
        if encoder_type == "transformer":
            encoder_config = {
                "vocab_size": 257,
                "d_model": 128,  # Reduced from 256
                "nhead": 4,  # Reduced from 8
                "num_layers": 2,  # Reduced from 4
                "dim_feedforward": 512,  # Reduced from 1024
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": use_cuda,
                "use_gradient_checkpointing": False,  # Can be enabled for memory saving
            }
        else:  # cnn
            encoder_config = {
                "vocab_size": 257,
                "embedding_dim": 128,
                "num_filters": 128,
                "filter_sizes": (3, 4, 5),
                "dropout": 0.1,
                "use_cuda": use_cuda,
            }

    # Set hidden_dim if not provided
    if hidden_dim <= 0:
        hidden_dim = 128  # Default

    # Set max_parameters if not provided
    if max_parameters <= 0:
        max_parameters = 12  # Default

    logger.info("Creating ParameterPredictionModel with:")
    logger.info(f"  Encoder type: {encoder_type}")
    logger.info(f"  Hidden dimension: {hidden_dim}")
    logger.info(f"  Max parameters: {max_parameters}")
    logger.info(f"  Type classes: {num_type_classes}")
    logger.info(f"  Function classes: {num_function_classes}")
    if encoder_config and encoder_type == "transformer":
        logger.info(
            f"  Transformer config: d_model={encoder_config.get('d_model', 'N/A')}, "
            f"nhead={encoder_config.get('nhead', 'N/A')}, "
            f"num_layers={encoder_config.get('num_layers', 'N/A')}, "
            f"dim_feedforward={encoder_config.get('dim_feedforward', 'N/A')}"
        )

    # Create model
    model = ParameterPredictionModel(
        num_type_classes=num_type_classes,
        max_parameters=max_parameters,
        encoder_type=encoder_type,
        encoder_config=encoder_config,
        hidden_dim=hidden_dim,
        dropout=0.2,
        use_function_conditioning=True,
        num_function_classes=num_function_classes,
        use_cuda=use_cuda,
    )

    # Compute and set class weights for type prediction if enabled
    if use_class_weights and train_samples is not None:
        logger.info("  Computing class weights for type prediction...")
        type_weights = dataset.compute_type_class_weights(train_samples)
        type_weights_tensor = torch.FloatTensor(type_weights)
        model.set_type_class_weights(type_weights_tensor)
        logger.info(
            f"    Type class weights: min={type_weights.min():.2f}, max={type_weights.max():.2f}"
        )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Create trainer
    trainer = ParameterPredictionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        use_cuda=use_cuda,
        checkpoint_dir=checkpoint_dir,
        model_name="parameter_predictor",
    )

    # Train model
    trainer.train(
        num_epochs=num_epochs,
        patience=min(10, num_epochs // 2),
        save_best=True,
    )

    # Calibrate temperatures on validation set (if available)
    if len(val_loader) > 0:
        logger.info("\nCalibrating temperature parameters...")
        model.calibrate_temperatures(val_loader, lr=0.01, max_iters=20)
    else:
        logger.info("\nSkipping temperature calibration (no validation data)")

    # Save final model
    model_path = Path(checkpoint_dir) / "parameter_predictor_final.pth"
    model.save_model(str(model_path))

    return model


def evaluate_models(
    function_model: FunctionNameClassifier,
    parameter_model: ParameterPredictionModel,
    test_loader: DataLoader,
    dataset: ParameterDataset,
    use_cuda: bool = True,
):
    """Evaluate trained models on test set.

    Args:
        function_model: Trained function name classifier
        parameter_model: Trained parameter prediction model
        test_loader: Test data loader
        dataset: ParameterDataset for mappings
        use_cuda: Whether to use GPU
    """
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Models on Test Set")
    logger.info("=" * 60)

    # Set models to evaluation mode
    function_model.eval()
    parameter_model.eval()

    # Evaluation metrics
    function_metrics = {
        "total": 0,
        "correct": 0,
        "top3_correct": 0,
        "confidences": [],
    }

    parameter_metrics = {
        "count_total": 0,
        "count_correct": 0,
        "type_total": 0,
        "type_correct": 0,
        "mask_total": 0,
        "mask_correct": 0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if use_cuda:
                batch = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            bytecode_tokens = batch["bytecode_tokens"]
            attention_mask = batch.get("attention_mask")
            true_function_idx = batch["function_name_idx"]
            true_param_count = batch["parameter_count"]
            true_param_types = batch["parameter_types"]
            true_param_mask = batch["parameter_mask"]

            # 1. Evaluate function name classifier
            function_outputs = function_model.predict(
                bytecode_tokens, attention_mask, threshold=0.0
            )
            func_preds = function_outputs["predictions"]
            func_confs = function_outputs["confidences"]
            func_probs = function_outputs["probs"]

            for i in range(len(func_preds)):
                function_metrics["total"] += 1
                if func_preds[i] == true_function_idx[i].item():
                    function_metrics["correct"] += 1
                top3 = torch.topk(func_probs[i], k=min(3, func_probs.size(-1)))
                if true_function_idx[i].item() in top3.indices.cpu().tolist():
                    function_metrics["top3_correct"] += 1
                function_metrics["confidences"].append(float(func_confs[i]))

            # 2. Evaluate parameter prediction model
            # predict() takes (bytecode_features, attention_mask, ...thresholds);
            # it does not accept function ids. Pass discriminating features so
            # evaluation matches training (avoids train/serve skew).
            param_outputs = parameter_model.predict(
                bytecode_tokens,
                attention_mask,
                type_threshold=0.0,
                mask_threshold=0.5,
                discriminating_features=batch.get("discriminating_features"),
            )

            # predict() returns per-sample "predictions" plus a "raw_outputs"
            # dict of batched tensors; use the latter for vectorised metrics.
            raw_outputs = param_outputs["raw_outputs"]
            type_preds = raw_outputs["predicted_types"]            # (B, max_params)
            mask_preds = (raw_outputs["mask_probs"] > 0.5).long()  # (B, max_params)

            batch_size = type_preds.shape[0]
            for i in range(batch_size):
                parameter_metrics["count_total"] += 1

                pred_count = int(mask_preds[i].sum())
                true_count = min(
                    true_param_count[i].item(), parameter_model.max_parameters
                )
                if pred_count == true_count:
                    parameter_metrics["count_correct"] += 1

                for pos in range(parameter_model.max_parameters):
                    if (
                        pos < true_param_types.shape[1]
                        and true_param_mask[i, pos].item() > 0.5
                    ):
                        parameter_metrics["type_total"] += 1
                        true_type = true_param_types[i, pos].item()
                        pred_type = int(type_preds[i, pos])
                        if pred_type == true_type:
                            parameter_metrics["type_correct"] += 1

                for pos in range(parameter_model.max_parameters):
                    if pos < true_param_mask.shape[1]:
                        parameter_metrics["mask_total"] += 1
                        true_mask = true_param_mask[i, pos].item() > 0.5
                        pred_mask = bool(mask_preds[i, pos])
                        if true_mask == pred_mask:
                            parameter_metrics["mask_correct"] += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Calculate metrics
    function_accuracy = (
        function_metrics["correct"] / function_metrics["total"] * 100
        if function_metrics["total"] > 0
        else 0.0
    )
    function_top3_accuracy = (
        function_metrics["top3_correct"] / function_metrics["total"] * 100
        if function_metrics["total"] > 0
        else 0.0
    )
    avg_confidence = (
        np.mean(function_metrics["confidences"])
        if function_metrics["confidences"]
        else 0.0
    )

    count_accuracy = (
        parameter_metrics["count_correct"] / parameter_metrics["count_total"] * 100
        if parameter_metrics["count_total"] > 0
        else 0.0
    )
    type_accuracy = (
        parameter_metrics["type_correct"] / parameter_metrics["type_total"] * 100
        if parameter_metrics["type_total"] > 0
        else 0.0
    )
    mask_accuracy = (
        parameter_metrics["mask_correct"] / parameter_metrics["mask_total"] * 100
        if parameter_metrics["mask_total"] > 0
        else 0.0
    )

    # Print results
    logger.info("\nFunction Name Classifier Results:")
    logger.info(f"  Accuracy: {function_accuracy:.2f}%")
    logger.info(f"  Top-3 Accuracy: {function_top3_accuracy:.2f}%")
    logger.info(f"  Average Confidence: {avg_confidence:.4f}")

    logger.info("\nParameter Prediction Model Results:")
    logger.info(f"  Count Accuracy: {count_accuracy:.2f}%")
    logger.info(f"  Type Accuracy: {type_accuracy:.2f}%")
    logger.info(f"  Mask Accuracy: {mask_accuracy:.2f}%")

    return {
        "function_accuracy": function_accuracy,
        "function_top3_accuracy": function_top3_accuracy,
        "avg_confidence": avg_confidence,
        "count_accuracy": count_accuracy,
        "type_accuracy": type_accuracy,
        "mask_accuracy": mask_accuracy,
    }


def main():
    """Main training function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train ML models for ABI reconstruction"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum number of training samples",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--train_function", action="store_true", help="Train function name classifier"
    )
    parser.add_argument(
        "--train_parameter",
        action="store_true",
        help="Train parameter prediction model",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate trained models"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="transformer",
        choices=["transformer", "cnn"],
        help="Type of encoder to use (default: transformer)",
    )
    parser.add_argument(
        "--encoder_config",
        type=str,
        default="",
        help='JSON string for encoder configuration (e.g., \'{"d_model": 64, "nhead": 2}\')',
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disable caching of tokenized bytecode (saves memory)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=-1,
        help="Hidden dimension for classifier head (default: auto)",
    )
    parser.add_argument(
        "--max_parameters",
        type=int,
        default=-1,
        help="Maximum number of parameters to predict (default: 12)",
    )
    parser.add_argument(
        "--no_filter_noisy",
        action="store_true",
        help="Disable noisy sample filtering",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class weighting for type prediction",
    )

    args = parser.parse_args()

    # Use CUDA if available and not disabled
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parse encoder_config if provided
    encoder_config = None
    if args.encoder_config:
        try:
            import json

            encoder_config = json.loads(args.encoder_config)
            # Ensure use_cuda is set in encoder config
            encoder_config["use_cuda"] = use_cuda
            logger.info(f"Using custom encoder config: {encoder_config}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse encoder_config JSON: {e}")
            logger.info("Using default encoder config instead.")

    # Prepare datasets
    train_loader, val_loader, test_loader, dataset, train_samples = prepare_datasets(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        disable_cache=args.disable_cache,
        filter_noisy=not args.no_filter_noisy,
    )

    # Train function name classifier
    function_model = None
    if args.train_function:
        function_model = train_function_name_classifier(
            train_loader,
            val_loader,
            dataset,
            use_cuda,
            args.checkpoint_dir,
            args.epochs,
            encoder_type=args.encoder_type,
            encoder_config=encoder_config,
            hidden_dim=args.hidden_dim,
        )

    # Train parameter prediction model
    parameter_model = None
    if args.train_parameter:
        parameter_model = train_parameter_prediction_model(
            train_loader,
            val_loader,
            dataset,
            use_cuda,
            args.checkpoint_dir,
            args.epochs,
            encoder_type=args.encoder_type,
            encoder_config=encoder_config,
            hidden_dim=args.hidden_dim,
            max_parameters=args.max_parameters,
            train_samples=train_samples,
            use_class_weights=not args.no_class_weights,
        )

    # Evaluate models
    if args.evaluate:
        # Load models if not trained
        if function_model is None:
            function_model_path = (
                Path(args.checkpoint_dir) / "function_classifier_final.pth"
            )
            if function_model_path.exists():
                function_model = FunctionNameClassifier.load_model(
                    str(function_model_path), use_cuda=use_cuda
                )
            else:
                logger.info("Function model not found. Please train it first.")
                return

        if parameter_model is None:
            parameter_model_path = (
                Path(args.checkpoint_dir) / "parameter_predictor_final.pth"
            )
            if parameter_model_path.exists():
                parameter_model = ParameterPredictionModel.load_model(
                    str(parameter_model_path), use_cuda=use_cuda
                )
            else:
                logger.info("Parameter model not found. Please train it first.")
                return

        # Evaluate
        evaluate_models(function_model, parameter_model, test_loader, dataset, use_cuda)

    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
