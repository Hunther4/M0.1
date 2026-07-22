"""Backend selection and safe fallback for scaled dot-product attention."""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
import warnings

import torch
import torch.nn.functional as F


class AttentionBackend(str, Enum):
    AUTO = "auto"
    FLASH = "flash"
    EFFICIENT = "efficient"
    MATH = "math"


class AttentionDispatcher:
    """Dispatch SDPA to a preferred backend and fall back to math safely."""

    def __init__(self, preferred_backend: str | AttentionBackend = AttentionBackend.AUTO) -> None:
        try:
            self.preferred_backend = AttentionBackend(preferred_backend)
        except ValueError as exc:
            choices = ", ".join(item.value for item in AttentionBackend)
            raise ValueError(f"Unknown attention backend {preferred_backend!r}; use {choices}") from exc
        self._warned = False

    def _context(self, backend: AttentionBackend):
        if backend is AttentionBackend.AUTO:
            return nullcontext()
        attention = getattr(torch.nn, "attention", None)
        if attention is None or not hasattr(attention, "sdpa_kernel"):
            return nullcontext()
        mapping = {
            AttentionBackend.FLASH: attention.SDPBackend.FLASH_ATTENTION,
            AttentionBackend.EFFICIENT: attention.SDPBackend.EFFICIENT_ATTENTION,
            AttentionBackend.MATH: attention.SDPBackend.MATH,
        }
        return attention.sdpa_kernel(mapping[backend])

    def dispatch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        try:
            with self._context(self.preferred_backend):
                return F.scaled_dot_product_attention(q, k, v, **kwargs)
        except RuntimeError as exc:
            if self.preferred_backend in (AttentionBackend.AUTO, AttentionBackend.MATH):
                raise
            if not self._warned:
                warnings.warn(
                    f"SDPA backend {self.preferred_backend.value!r} failed; falling back to math: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            with self._context(AttentionBackend.MATH):
                return F.scaled_dot_product_attention(q, k, v, **kwargs)
