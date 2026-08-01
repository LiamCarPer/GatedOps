"""Unit tests for the synthetic data generator and its hashing."""

from gatedops.data.synthetic import data_hash, generate_churn


def test_generator_is_deterministic() -> None:
    first = generate_churn(n_rows=500, seed=7)
    second = generate_churn(n_rows=500, seed=7)

    assert first.equals(second)
    assert data_hash(first) == data_hash(second)


def test_data_hash_changes_with_seed() -> None:
    first = generate_churn(n_rows=500, seed=7)
    second = generate_churn(n_rows=500, seed=8)

    assert data_hash(first) != data_hash(second)


def test_signal_strength_drives_separability() -> None:
    good = generate_churn(n_rows=4000, signal_strength=0.9, seed=1)
    bad = generate_churn(n_rows=4000, signal_strength=0.1, seed=1)

    for frame in (good, bad):
        assert 0.1 < frame["churn"].mean() < 0.9

    good_corr = abs(good["engagement_score"].corr(good["churn"]))
    bad_corr = abs(bad["engagement_score"].corr(bad["churn"]))

    assert good_corr > bad_corr
