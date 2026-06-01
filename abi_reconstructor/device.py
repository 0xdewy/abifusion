"""Device management utilities for consistent GPU/CPU handling.

This module provides consistent device management across the ABI Reconstructor:
- get_default_device: Returns CUDA if available, else CPU
- DeviceManager: Context manager for consistent device handling
- graceful_fallback: Automatic CPU fallback when CUDA fails
- batch_device_aware: Efficient device management for batches
"""

import gc
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class DeviceError(Exception):
    """Raised when device operations fail."""
    pass


class CUDANotAvailableError(DeviceError):
    """Raised when CUDA is requested but not available."""
    pass


class DeviceMismatchError(DeviceError):
    """Raised when there's a device mismatch."""
    pass


def get_default_device() -> torch.device:
    """Get the default device (CUDA if available, else CPU).

    Returns:
        torch.device: The default device for tensor operations
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_for_cuda_flag(use_cuda: bool) -> torch.device:
    """Get device based on use_cuda flag.

    Args:
        use_cuda: Whether CUDA is requested

    Returns:
        torch.device: The appropriate device
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_device_available(device: torch.device) -> bool:
    """Check if a device is available.

    Args:
        device: Device to check

    Returns:
        True if device is available
    """
    if device.type == "cuda":
        return torch.cuda.is_available()
    return True


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "default_device": str(get_default_device()),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_mb": torch.cuda.get_device_properties(i).total_memory / (1024 * 1024),
            }
            info["devices"].append(device_info)

    return info


def to_device(
    data: Union[torch.Tensor, nn.Module],
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, nn.Module]:
    """Move a tensor or module to the specified device.

    Args:
        data: Tensor or module to move
        device: Target device (uses CPU if None)

    Returns:
        Data moved to the target device
    """
    if device is None:
        device = torch.device("cpu")
    return data.to(device)


class DeviceManager:
    """Context manager and utility class for consistent device handling.

    Provides:
    - Automatic device detection and fallback
    - Context managers for temporary device changes
    - Batch-aware device operations
    - Memory management utilities

    Example:
        >>> device_manager = DeviceManager()
        >>> model = BytecodeFeatureExtractor()
        >>> model = device_manager.move_model_to_device(model)
        >>> # or use context manager
        >>> with device_manager.use_device(torch.device("cpu")):
        ...     output = model(input_ids)
        >>> # device_manager is restored after context
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        enable_cuda: bool = True,
        memory_fraction: Optional[float] = None
    ):
        """Initialize DeviceManager.

        Args:
            device: Specific device to use (auto-detects if None)
            enable_cuda: Whether to enable CUDA (set False to force CPU)
            memory_fraction: Fraction of GPU memory to use (None = all)
        """
        self._enable_cuda = enable_cuda

        if device is None:
            if enable_cuda and torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        else:
            if device.type == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA device requested but not available, using CPU",
                    stacklevel=2,
                )
                self._device = torch.device("cpu")
            else:
                self._device = device

        self._memory_fraction = memory_fraction
        if self._device.type == "cuda" and memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self._device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self._device.type == "cpu"

    def move_model_to_device(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Move a model to the device.

        Args:
            model: PyTorch model to move
            device: Target device (uses default if None)

        Returns:
            Model moved to device
        """
        target = device or self._device
        return model.to(target)

    def to_device(
        self,
        data: Union[torch.Tensor, nn.Module]
    ) -> Union[torch.Tensor, nn.Module]:
        """Move data to the device.

        Args:
            data: Tensor or module to move

        Returns:
            Data on the target device
        """
        return to_device(data, self._device)

    @contextmanager
    def use_device(self, device: torch.device):
        """Context manager to temporarily change the device.

        Example:
            >>> dm = DeviceManager()
            >>> with dm.use_device(torch.device("cpu")):
            ...     # code using dm.device is now cpu
            >>> # dm.device is restored after context
        """
        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA device requested but not available, using CPU",
                stacklevel=2,
            )
            device = torch.device("cpu")

        original_device = self._device
        self._device = device
        try:
            yield self._device
        finally:
            self._device = original_device

    @contextmanager
    def use_cpu(self):
        """Context manager to temporarily use CPU.

        Example:
            >>> dm = DeviceManager()
            >>> with dm.use_cpu():
            ...     # code using CPU
        """
        with self.use_device(torch.device("cpu")):
            yield

    @contextmanager
    def use_cuda_if_available(self):
        """Context manager to use CUDA if available, else CPU.

        Example:
            >>> dm = DeviceManager()
            >>> with dm.use_cuda_if_available():
            ...     # code using best available device
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        with self.use_device(device):
            yield

    def get_device_for_data(self, data: torch.Tensor) -> torch.device:
        """Get the appropriate device for given data.

        If data is already on a device, returns that device.
        Otherwise returns the default.

        Args:
            data: Input tensor

        Returns:
            Device for the data
        """
        if isinstance(data, torch.Tensor):
            return data.device
        return self._device

    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            gc.collect()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage.

        Returns:
            Dictionary with memory usage statistics
        """
        usage = {"device": str(self._device)}

        if self.is_cuda:
            usage["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            usage["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            usage["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        else:
            import psutil
            process = psutil.Process()
            usage["ram_used_mb"] = process.memory_info().rss / (1024 * 1024)

        return usage

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device}, enable_cuda={self._enable_cuda})"


_default_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get the global default DeviceManager instance.

    Returns:
        Global DeviceManager instance
    """
    global _default_device_manager
    if _default_device_manager is None:
        _default_device_manager = DeviceManager()
    return _default_device_manager


def set_default_device_manager(manager: DeviceManager):
    """Set the global default DeviceManager instance.

    Args:
        manager: DeviceManager to set as default
    """
    global _default_device_manager
    _default_device_manager = manager


def batch_to_device(
    batch: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Move all tensors in a batch to the specified device.

    Args:
        batch: Dictionary containing tensors
        device: Target device (uses default if None)

    Returns:
        Batch with all tensors moved to device
    """
    if device is None:
        device = get_default_device()

    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
