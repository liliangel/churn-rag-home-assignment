import numpy as np
import pandas as pd

def generate_insurance_panel(
    n_policies: int = 2000,
    months: int = 12,
    start: str = "2023-01",
    drift_start: str = "2023-07",
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    month_index = pd.period_range(start=start, periods=months, freq="M")
    drift_start_p = pd.Period(drift_start, freq="M")

    # --- policy-level attributes ---
    policy_id = np.arange(1, n_policies + 1)

    region = rng.choice(["NORTH", "SOUTH", "EAST", "WEST"], size=n_policies)
    has_agent = rng.binomial(1, 0.55, size=n_policies)
    is_smoker = rng.binomial(1, 0.22, size=n_policies)
    dependents = rng.choice([0, 1, 2, 3, 4], size=n_policies, p=[0.35, 0.28, 0.22, 0.10, 0.05])

    age0 = rng.integers(18, 76, size=n_policies)
    tenure0 = rng.integers(0, 121, size=n_policies)  # 0..120 months

    # Coverage: lognormal-ish distribution
    coverage0 = np.round(np.exp(rng.normal(np.log(20000), 0.6, size=n_policies)) / 100) * 100
    coverage0 = np.clip(coverage0, 5000, 200000)

    # Build full panel grid
    df = pd.MultiIndex.from_product(
        [policy_id, month_index],
        names=["policy_id", "month"]
    ).to_frame(index=False)

    # Map policy-level fields into panel
    pol = pd.DataFrame({
        "policy_id": policy_id,
        "region": region,
        "has_agent": has_agent,
        "is_smoker": is_smoker,
        "dependents": dependents,
        "age0": age0,
        "tenure0": tenure0,
        "coverage0": coverage0
    })
    df = df.merge(pol, on="policy_id", how="left")

    # Sort and make integer month index per policy: 0..months-1
    df = df.sort_values(["policy_id", "month"]).reset_index(drop=True)
    df["m_idx"] = df.groupby("policy_id").cumcount()

    # Features over time
    df["age"] = df["age0"] + (df["m_idx"] // 12)  # within 12 months basically stable
    df["tenure_m"] = df["tenure0"] + df["m_idx"]

    # Risk score (pre-drift) - intuitive components
    region_risk = df["region"].map({"NORTH": 0.05, "SOUTH": 0.08, "EAST": 0.06, "WEST": 0.07}).astype(float)
    smoker_risk = df["is_smoker"] * 0.10
    dep_risk = df["dependents"] * 0.015
    agent_protect = df["has_agent"] * (-0.03)
    tenure_protect = -0.0008 * df["tenure_m"]  # longer tenure -> less lapse

    # Premium correlated with coverage and risk
    base_rate = 0.0045  # ~0.45% monthly of coverage
    noise = rng.normal(0, 8, size=len(df))

    df["coverage"] = df["coverage0"]
    df["premium"] = (
        df["coverage"] * base_rate
        * (1 + 0.25*df["is_smoker"] + 0.06*df["dependents"] + 0.08*(df["region"].isin(["SOUTH", "WEST"]).astype(int)))
        * (1 - 0.05*df["has_agent"])
    ) + noise
    df["premium"] = np.round(np.clip(df["premium"], 20, None), 2)

    # --- drift after drift_start: e.g., economic shock increases lapse propensity ---
    df["post_drift"] = (df["month"] >= drift_start_p).astype(int)

    # Monthly lapse probability (hazard-like)
    prem_to_cov = df["premium"] / (df["coverage"] / 1000.0)  # premium per $1k coverage
    x = (
        -3.0
        + 0.18 * prem_to_cov
        + 1.1 * smoker_risk
        + 0.7 * region_risk
        + 0.5 * dep_risk
        + 0.6 * df["post_drift"]      # drift increases lapse risk after 2023-07
        + agent_protect
        + tenure_protect
        + rng.normal(0, 0.25, size=len(df))
    )
    p_lapse_month = 1 / (1 + np.exp(-x))
    p_lapse_month = np.clip(p_lapse_month, 0.001, 0.30)

    # Simulate first lapse month per policy (once lapsed -> stays lapsed)
    df["lapsed_this_month"] = 0

    for pid, idxs in df.groupby("policy_id").groups.items():
        lapsed = False
        for i in idxs:
            if lapsed:
                continue
            if rng.random() < p_lapse_month[i]:
                df.at[i, "lapsed_this_month"] = 1
                lapsed = True

    # Mark active status (after lapse month -> inactive)
    df["is_active"] = 1
    for pid, g in df.groupby("policy_id"):
        lapse_rows = g.index[g["lapsed_this_month"] == 1].tolist()
        if lapse_rows:
            lapse_i = lapse_rows[0]
            after = g.index[g.index > lapse_i]
            df.loc[after, "is_active"] = 0

    # --- Target: lapse during next 3 months ---
    df["lapse_next_3m"] = 0
    for pid, g in df.groupby("policy_id"):
        g = g.sort_values("month")
        lapse_pos = g.index[g["lapsed_this_month"] == 1].tolist()
        if not lapse_pos:
            continue
        lapse_i = lapse_pos[0]
        lapse_m_idx = df.at[lapse_i, "m_idx"]
        prior_mask = (g["m_idx"] >= lapse_m_idx - 3) & (g["m_idx"] <= lapse_m_idx - 1)
        df.loc[g.index[prior_mask], "lapse_next_3m"] = 1

    # --- Leakage trap feature (uses FUTURE info on purpose) ---
    # Future lapse indicators (these look into the future -> leakage)
    g = df.groupby("policy_id", group_keys=False)
    df["will_lapse_in_1m"] = g["lapsed_this_month"].shift(-1).fillna(0).astype(int)
    df["will_lapse_in_2m"] = g["lapsed_this_month"].shift(-2).fillna(0).astype(int)
    df["will_lapse_in_3m"] = g["lapsed_this_month"].shift(-3).fillna(0).astype(int)

    # Leakage trap: incorrectly "known today" but derived from future
    df["post_event_leak"] = (
        df["will_lapse_in_1m"] | df["will_lapse_in_2m"] | df["will_lapse_in_3m"]
    ).astype(int)

    # Optional: keep only the trap, drop helper future indicators
    df = df.drop(columns=["will_lapse_in_1m", "will_lapse_in_2m", "will_lapse_in_3m"])

    # Clean up original policy-level helpers
    df = df.drop(columns=["age0", "tenure0", "coverage0"])

    # Convert month to string "YYYY-MM" for easy CSV saving
    df["month"] = df["month"].astype(str)

    return df



    
def strict_time_split(df):
    """
    Perform a strict temporal split of the panel dataset into
    train / validation / test sets.

    Why temporal split?
    -------------------
    - Policies evolve over time.
    - Future information must NEVER be used to train a model.
    - Therefore: the model trains on early months and is evaluated
      on later, unseen months — just like real forecasting.

    Split logic:
        • Train = first 8 months
        • Val   = months 9 and 10
        • Test  = months 11 and 12
    """

    # Name of the prediction target
    TARGET = "lapse_next_3m"

    # Columns that should NEVER be used as features:
    #   - TARGET: the label we want to predict
    #   - policy_id, month: identifiers (not predictive)
    #   - lapsed_this_month, is_active: post-event indicators / state variables
    DROP_COLS = [
        TARGET,
        "policy_id",
        "month",
        "lapsed_this_month",
        "is_active"
    ]

    # Leakage columns: anything that starts with
    # "post_event_" contains FUTURE information by construction.
    # These must be EXCLUDED from training.
    LEAK_COLS = [c for c in df.columns if c.startswith("post_event_")]

    # Final feature set: all columns except target, identifiers,
    # state variables, and leakage traps.
    FEATURES = [c for c in df.columns if c not in DROP_COLS + LEAK_COLS]

    # Work on a copy to avoid modifying the original df
    df = df.copy()

    # Sorted list of all unique month values
    months = sorted(df["month"].unique())

    # Temporal split:
    train_months = months[:8]   # months 1–8
    val_months   = months[8:10] # months 9–10
    test_months  = months[10:]  # months 11–12

    # Filter rows belonging to each period
    train = df[df["month"].isin(train_months)]
    val   = df[df["month"].isin(val_months)]
    test  = df[df["month"].isin(test_months)]

    # Separate features and targets
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val,   y_val   = val[FEATURES],   val[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    return X_train, y_train, X_val, y_val, X_test, y_test

