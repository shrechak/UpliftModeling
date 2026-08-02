import numpy as np
from scipy.special import expit as sigmoid


def generate_campaign_data(n=20_000, seed=42):
    """
    Synthetic food delivery campaign data with known ground-truth uplift.

    Returns a dict with keys:
        X            : (n, 15) feature matrix
        feature_names: list of 15 column names
        T            : binary treatment assignment (50/50 randomised)
        Y            : binary outcome (placed order)
        tau          : ground-truth individual treatment effect (never observable in production)
        base_p       : counterfactual order probability without voucher
        churn_risk   : stylised churn risk score (structurally independent of tau by design)
    """
    rng = np.random.default_rng(seed)

    # ── Customer features ────────────────────────────────────────────────────
    days_since_last_order     = rng.exponential(scale=14, size=n).clip(1, 90)
    orders_last_30d           = rng.poisson(lam=3, size=n).clip(0, 20).astype(float)
    orders_last_90d           = orders_last_30d * 3 + rng.poisson(1, n)
    avg_order_value           = rng.lognormal(mean=3.2, sigma=0.5, size=n)
    voucher_use_rate          = rng.beta(2, 5, size=n)
    voucher_orders_last_90d   = (orders_last_90d * voucher_use_rate).astype(int).astype(float)
    distinct_cuisines_30d     = rng.poisson(2, n).clip(1, 8).astype(float)
    support_tickets_open      = rng.poisson(0.2, n).clip(0, 3).astype(float)
    days_since_bad_experience = rng.exponential(30, n).clip(1, 180)
    tenure_days               = rng.exponential(180, n).clip(7, 1000)
    session_days_last_14d     = rng.poisson(4, n).clip(0, 14).astype(float)
    avg_delivery_time         = rng.normal(32, 8, n).clip(15, 60)
    refund_rate               = rng.beta(1, 10, n)
    email_open_rate_30d       = rng.beta(2, 5, n)
    push_open_rate_30d        = rng.beta(1, 6, n)

    feature_names = [
        "days_since_last_order", "orders_last_30d", "orders_last_90d",
        "avg_order_value", "voucher_use_rate", "voucher_orders_last_90d",
        "distinct_cuisines_30d", "support_tickets_open", "days_since_bad_experience",
        "tenure_days", "session_days_last_14d", "avg_delivery_time",
        "refund_rate", "email_open_rate_30d", "push_open_rate_30d",
    ]

    X = np.column_stack([
        days_since_last_order, orders_last_30d, orders_last_90d,
        avg_order_value, voucher_use_rate, voucher_orders_last_90d,
        distinct_cuisines_30d, support_tickets_open, days_since_bad_experience,
        tenure_days, session_days_last_14d, avg_delivery_time,
        refund_rate, email_open_rate_30d, push_open_rate_30d,
    ])

    # ── Ground-truth uplift τ ────────────────────────────────────────────────
    # Persuadables: moderately lapsed, price-sensitive, email-reachable
    persuadable = (
        0.20 * sigmoid((days_since_last_order - 10) / 5)
              * sigmoid(-(days_since_last_order - 25) / 5)
        + 0.25 * voucher_use_rate
        + 0.10 * email_open_rate_30d
    )
    # Sleeping dogs: very frequent + long-tenured → voucher trains discount expectation
    sleeping_dog = (
        0.20 * sigmoid((orders_last_30d - 7) / 2)
        + 0.15 * sigmoid((tenure_days - 250) / 50)
    )
    tau = (persuadable - sleeping_dog + rng.normal(0, 0.03, n)).clip(-0.35, 0.40)

    # ── Randomised treatment ─────────────────────────────────────────────────
    T = rng.binomial(1, 0.5, n)

    # ── Outcomes ─────────────────────────────────────────────────────────────
    base_p = sigmoid(
        -1.0
        - 0.04 * days_since_last_order
        + 0.15 * orders_last_30d
        + 0.10 * session_days_last_14d
        - 0.20 * support_tickets_open
    )
    p = np.clip(base_p + T * tau, 0.01, 0.99)
    Y = rng.binomial(1, p)

    # ── Churn risk score (structurally independent of τ by design) ───────────
    churn_risk = sigmoid(
         0.05 * days_since_last_order
        - 0.12 * orders_last_30d
        - 0.08 * session_days_last_14d
        + 0.25 * support_tickets_open
        + rng.normal(0, 0.05, n)
    )

    return dict(
        X=X,
        feature_names=feature_names,
        T=T,
        Y=Y,
        tau=tau,
        base_p=base_p,
        churn_risk=churn_risk,
    )
