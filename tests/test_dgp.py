import numpy as np
import pytest
from src.dgp import generate_campaign_data


@pytest.fixture(scope="module")
def data():
    return generate_campaign_data(n=20_000, seed=42)


def test_feature_shape(data):
    assert data["X"].shape == (20_000, 15)


def test_feature_names_length(data):
    assert len(data["feature_names"]) == 15


def test_sleeping_dog_rate(data):
    """~14% of customers should have negative uplift (sleeping dogs)."""
    rate = (data["tau"] < 0).mean()
    assert 0.10 <= rate <= 0.20, f"Sleeping dog rate {rate:.1%} outside [10%, 20%]"


def test_churn_tau_independence(data):
    """Churn risk and uplift are structurally independent — near-zero correlation by design."""
    corr = abs(np.corrcoef(data["churn_risk"], data["tau"])[0, 1])
    assert corr < 0.15, f"|Corr(churn, τ)| = {corr:.3f}, expected < 0.15"


def test_treatment_balance(data):
    """Treatment is a 50/50 coin flip — rate should be near 0.5."""
    rate = data["T"].mean()
    assert 0.47 <= rate <= 0.53, f"Treatment rate {rate:.3f} not near 0.5"


def test_tau_clipped(data):
    """τ is clipped to [-0.35, 0.40] in the DGP."""
    tau = data["tau"]
    assert tau.min() >= -0.36
    assert tau.max() <= 0.41


def test_outcome_binary(data):
    assert set(data["Y"]).issubset({0, 1})


def test_base_p_in_unit_interval(data):
    assert data["base_p"].min() >= 0.01
    assert data["base_p"].max() <= 0.99


def test_reproducibility():
    """Same seed must produce identical output."""
    d1 = generate_campaign_data(n=1_000, seed=0)
    d2 = generate_campaign_data(n=1_000, seed=0)
    np.testing.assert_array_equal(d1["tau"], d2["tau"])
    np.testing.assert_array_equal(d1["Y"],   d2["Y"])


def test_different_seeds_differ():
    """Different seeds must produce different data."""
    d1 = generate_campaign_data(n=1_000, seed=0)
    d2 = generate_campaign_data(n=1_000, seed=1)
    assert not np.array_equal(d1["tau"], d2["tau"])
