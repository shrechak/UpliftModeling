# Synthetic Data Generation — Design Notes

Full rationale behind the data-generating process (DGP) used in the companion notebook.  
The mechanics live in [`src/dgp.py`](../src/dgp.py). This document explains *why* each choice was made.

---

## Why synthetic data?

In a real campaign you never observe τ — the causal effect of the voucher on a specific customer. You only see what happened, not what *would have* happened without the voucher. Synthetic data lets you construct τ from scratch, so you can check whether the model recovers it.

The simulation has five components: customer features, ground-truth uplift τ, randomised treatment, observed outcomes, and a churn risk score as the incumbent benchmark.

---

## 1. Customer features (15 signals)

Features are drawn from distributions chosen to mimic a food delivery platform.

### Recency & frequency

| Feature | Distribution | Rationale |
|---|---|---|
| `days_since_last_order` | Exponential(scale=14), clip 1–90 | Exponential is the natural distribution for elapsed time since an event (continuous counterpart of the Poisson process). Scale=14 puts the mean lapse at 14 days — most customers are recent, with a long tail. Clipped at 90: beyond that they are effectively dormant. |
| `orders_last_30d` | Poisson(λ=3), clip 0–20 | Poisson is the standard for count-of-events in a fixed window. λ=3 gives ~3 orders per month for an active user. |
| `orders_last_90d` | `orders_last_30d × 3 + Poisson(1)` | Not independently drawn — derived from the 30-day count to preserve internal consistency. An independent draw could generate someone with 0 orders last month but 15 last quarter. The Poisson(1) add-on allows natural seasonal variation. |

### Monetary

| Feature | Distribution | Rationale |
|---|---|---|
| `avg_order_value` | LogNormal(μ=3.2, σ=0.5) | Spend distributions are always right-skewed: most orders are moderate value, a long tail of large orders. Log-normal is the standard model for monetary amounts. μ=3.2 in log space → median ≈ £25. |

### Voucher sensitivity

| Feature | Distribution | Rationale |
|---|---|---|
| `voucher_use_rate` | Beta(2, 5) | A rate bounded in [0,1] → Beta family. Beta(2,5) has mean ≈ 0.29 and is skewed toward lower values: most customers rarely use vouchers, a minority are heavy discount users. Realistic shape for price sensitivity. |
| `voucher_orders_last_90d` | `floor(orders_last_90d × voucher_use_rate)` | Derived, not independently sampled. Maintains consistency: a customer with low `voucher_use_rate` and few total orders cannot end up with many voucher orders. |

### Engagement

| Feature | Distribution | Rationale |
|---|---|---|
| `distinct_cuisines_30d` | Poisson(2), clip 1–8 | Count of distinct cuisine types — Poisson for count data. λ=2 means most customers order from 1–3 types per month. Hard clip at 1 (must have ordered at least once) and 8. |
| `session_days_last_14d` | Poisson(4), clip 0–14 | Days with at least one app session. Poisson because it's a count per fixed window. Hard upper clip at 14 since you cannot have more active days than days in the window. |

### Experience quality

| Feature | Distribution | Rationale |
|---|---|---|
| `support_tickets_open` | Poisson(0.2), clip 0–3 | A rare-event count — most customers have zero open tickets. λ=0.2 puts the vast majority at 0. Clip at 3: simultaneously holding more than 3 tickets is essentially impossible. |
| `days_since_bad_experience` | Exponential(30), clip 1–180 | Same elapsed-time logic as recency. Scale=30 means the average bad experience was ~30 days ago. Clip at 180: beyond 6 months a past incident is unlikely to drive current behaviour. |
| `avg_delivery_time` | Normal(32, 8), clip 15–60 | Delivery time is a physical process — sum of many small independent factors (kitchen time, rider proximity, traffic). Central Limit Theorem pushes this toward Normal. Mean 32 min, σ=8 gives realistic spread. Clip at 15 (physically impossible faster) and 60 (outlier/complaint territory). |
| `refund_rate` | Beta(1, 10) | A proportion bounded in [0,1] → Beta. Beta(1,10) has mean ≈ 0.09 and is heavily concentrated near zero: most customers essentially never request refunds, a small tail of habitual complainers. |

### Channel reachability

| Feature | Distribution | Rationale |
|---|---|---|
| `email_open_rate_30d` | Beta(2, 5) | Engagement rate in [0,1] → Beta. Mean ≈ 0.29, skewed low. Consistent with real-world email open rates (20–30% average, many customers near zero). |
| `push_open_rate_30d` | Beta(1, 6) | Same family but Beta(1,6) has mean ≈ 0.14 — lower than email. Push notification engagement tends to be weaker, and more customers fully suppress push notifications. |

**The general pattern:** use **Exponential** for elapsed time, **Poisson** for event counts in a window, **Log-normal** for monetary amounts, and **Beta** for any rate or proportion bounded in [0,1].

---

## 2. Ground-truth uplift τ

τ is the individual treatment effect — the causal change in order probability that a voucher *creates* for a specific customer. It has two components:

### Persuadable score (pushes τ upward)

```
0.20 × sigmoid((days_since − 10) / 5) × sigmoid(−(days_since − 25) / 5)
+ 0.25 × voucher_use_rate
+ 0.10 × email_open_rate_30d
```

The first term is a soft bell-shaped window over recency. Two opposing sigmoids multiply together: one switches *on* around day 10 (customer is starting to lapse), the other switches *off* around day 25 (too lapsed to recover). The product peaks in the 10–25 day window — absent enough to need a nudge, recent enough that one works.

The remaining terms add price sensitivity (`voucher_use_rate`) and channel reachability (`email_open_rate_30d`).

### Sleeping dog score (pushes τ downward)

```
0.20 × sigmoid((orders_last_30d − 7) / 2)
+ 0.15 × sigmoid((tenure_days − 250) / 50)
```

Very frequent orderers (>7/month) would place that order without a voucher — the discount just cannibalises margin. Long-tenured customers (>250 days) have learned to expect discounts; sending one reinforces dependency rather than creating new demand.

### Final τ

```
τ = clip(persuadable − sleeping_dog + N(0, 0.03), −0.35, 0.40)
```

Small Gaussian noise adds individual heterogeneity. Clipping keeps τ within a plausible range for a probability shift. Around 14% of customers end up with negative τ — the sleeping dogs.

---

## 3. Treatment assignment

```
T ~ Bernoulli(0.5)  — a coin flip per customer
```

This is the ingredient that real campaign logs cannot provide. Observational voucher data conflates *who was selected* with *who responded* — any model trained on it learns the selection rule, not the causal effect. With a 50/50 randomised split the treated and control groups are exchangeable by design, so any difference in outcomes is attributable to the voucher alone.

---

## 4. Outcomes

```
base_p = sigmoid(−1.0 − 0.04×days_since + 0.15×orders_30d + 0.10×session_days − 0.20×support)
p      = clip(base_p + T×τ, 0.01, 0.99)
Y      ~ Bernoulli(p)
```

`base_p` is each customer's order probability without a voucher — driven by recency, recent order frequency, session engagement, and open support tickets. When `T=1`, τ shifts that probability: upward for persuadables, downward for sleeping dogs. Y is then a binary draw at probability p.

---

## 5. Churn risk score (the incumbent model)

```
churn_risk = sigmoid(0.05×days_since − 0.12×orders_30d − 0.08×session_days + 0.25×support + ε)
```

A stylised version of the model every retention team already has. Driven by recency and activity — sensible signals for *who is likely to leave*, with almost no overlap with the drivers of τ (voucher sensitivity, moderate lapse window, email reachability).

This is why `Corr(churn_risk, τ) ≈ 0.075` — the post's central claim made falsifiable: **risk and responsiveness are independent dimensions**. A churn model cannot be repurposed as a targeting model.
