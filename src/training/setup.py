"""Device and stdout setup utilities for M0.1 training scripts.

Provides the shared boilerplate used at the top of every training script
to configure the compute device and output encoding.
"""

import sys

import torch


def setup_device():
    """Return the best available torch device and print GPU info.

    Checks for CUDA availability and returns the appropriate device. Prints
    the device name for logging purposes.

    Returns:
        torch.device: CUDA device if available, else CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device} (CPU)")
    return device


def setup_stdout():
    """Configure stdout for UTF-8 encoding if supported.

    Reconfigures sys.stdout to use UTF-8 encoding when the reconfigure
    method is available (Python 3.7+). This is needed because training
    scripts print Spanish text and emoji.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
