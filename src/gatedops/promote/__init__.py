"""Guarded model promotion.

Promotion is the point where a model becomes what production serves. It is
guarded so that only a gated, hash-verified model can reach that state, and it
is expressed against a small registry protocol so the policy can be enforced on
top of any registry (MLflow today, something else tomorrow).
"""

from gatedops.promote.promote import (
    ModelRegistry,
    PromoteBlockedError,
    PromoteReceipt,
    promote,
)

__all__ = ["ModelRegistry", "PromoteBlockedError", "PromoteReceipt", "promote"]
