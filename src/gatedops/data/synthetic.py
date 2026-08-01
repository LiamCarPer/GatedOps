"""Synthetic customer-churn data with a controllable signal.

The generator is deterministic for a given seed, so the same configuration
always produces the same dataset and therefore the same ``data_hash``. The
``signal_strength`` knob controls how strongly the features predict churn:
high values produce a model that comfortably beats the gate, low values produce
a model that must be blocked. That makes the gate demo controlled instead of
staged.
"""

import hashlib

import numpy as np
import pandas as pd

FEATURES = [
    "tenure_years",
    "monthly_spend",
    "support_tickets",
    "usage_frequency",
    "engagement_score",
    "has_contract",
    "payment_delay",
]

# Churn is driven by low tenure/engagement, no contract, more tickets and
# payment delays. Coefficients act on standardized features.
_COEFFICIENTS = np.array([-0.5, 0.4, 0.6, -0.4, -1.2, -1.0, 0.8])
_NOISE_SCALE = 1.0
_SIGNAL_GAIN = 3.0


def data_hash(frame: pd.DataFrame) -> str:
    """Stable content hash of a dataset, independent of file encoding."""
    canonical = frame.to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_churn(
    n_rows: int = 20_000,
    signal_strength: float = 0.8,
    seed: int = 42,
    target: str = "churn",
) -> pd.DataFrame:
    """Return a deterministic churn-style dataset.

    ``signal_strength`` scales the linear predictor before the noise term is
    added, so it is a direct handle on how separable the classes are.
    """
    rng = np.random.default_rng(seed)

    frame = pd.DataFrame(
        {
            "tenure_years": rng.lognormal(mean=1.2, sigma=0.7, size=n_rows),
            "monthly_spend": rng.lognormal(mean=4.0, sigma=0.5, size=n_rows),
            "support_tickets": rng.poisson(lam=1.5, size=n_rows).astype(int),
            "usage_frequency": rng.normal(loc=50.0, scale=20.0, size=n_rows),
            "engagement_score": rng.uniform(0.0, 1.0, size=n_rows),
            "has_contract": rng.integers(0, 2, size=n_rows),
            "payment_delay": rng.exponential(scale=5.0, size=n_rows),
        }
    )

    standardized = (frame - frame.mean()) / frame.std()
    logit = (
        standardized.values @ _COEFFICIENTS * signal_strength * _SIGNAL_GAIN
        + rng.normal(0.0, _NOISE_SCALE, size=n_rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    frame[target] = rng.binomial(1, probability).astype(int)
    return frame
