# DISCUSSION

## 1. Synthetic Data Generation

### Design Goals
The synthetic dataset was designed to resemble a realistic **monthly insurance policy panel** while remaining fully controllable and leakage-aware. The primary goals were:
- Temporal structure (policies evolving month by month)
- Heterogeneous customer risk profiles
- Intuitive relationships between premium, coverage, tenure, and lapse risk
- Explicit control over drift and post-event behavior

### Key Challenges & Decisions

**a) Temporal consistency**  
A common pitfall in synthetic churn data is unintentionally leaking future information. This was avoided by:
- Assigning a fixed monthly index per policy (`m_idx`)
- Ensuring all features at month *t* depend only on information available at or before *t*
- Simulating at most one lapse event per policy and marking the policy inactive afterward

**b) Realistic lapse mechanism**  
Instead of random labels, lapse events were generated using a smooth, hazard-style probability driven by:
- Premium-to-coverage ratio
- Smoker status
- Number of dependents
- Region
- Tenure (protective effect)
- Agent presence (protective effect)

This produces a target that is **learnable but noisy**, closely resembling real churn problems.

**c) Concept drift**  
A simple but explicit drift was introduced after `2023-07` (`post_drift`) to simulate macro changes (e.g., economic pressure). This tests whether the model generalizes across time rather than memorizing a static distribution.

**d) Leakage trap**  
A deliberate post-event feature (`post_event_leak`) was introduced that encodes future lapse information. This feature perfectly predicts the target and was **explicitly excluded from training**. Its sole purpose is to demonstrate awareness of temporal leakage and causality constraints.

---

## 2. Target Definition

The target `lapse_next_3m` is defined as:

> For each policy-month *t*, the target equals 1 if the policy lapses in any of months *t+1, t+2, or t+3*.

This formulation reflects a realistic **early-warning scenario**, enabling proactive retention actions before an actual lapse occurs.

---

## 3. Data Splitting Strategy

A **strict time-based split** was used:
- Training on earlier months
- Validation on subsequent months
- Testing on the latest months

Policies are allowed to appear across splits **at different points in time**, preserving historical continuity. Random splits were intentionally avoided to prevent temporal leakage.

---

## 4. Model Choice & Training

### Model
A gradient-boosted tree model (XGBoost) was selected because it:
- Performs strongly on tabular data
- Captures non-linear interactions
- Trains efficiently
- Supports early stopping and SHAP-based explainability

Light hyperparameter tuning (≤30 Optuna trials) was used to balance performance with simplicity.

### Early Stopping
Early stopping was applied using a validation set to reduce overfitting, which is particularly important in time-dependent settings with drift.

---

## 5. Evaluation Metrics

### Primary Metric: AUC-PR
AUC-PR was chosen as the primary metric because lapse is a **rare event**, and precision-recall metrics are more informative than ROC-AUC under class imbalance.

### Business-Oriented Metrics
Precision@1% and Precision@5% were reported to reflect operational constraints where only a small subset of customers can be contacted proactively.

---

## 6. Results Interpretation

The model achieves:
- A meaningful AUC-PR, indicating effective risk ranking
- Improved precision at top percentiles, aligning with real retention workflows

Performance is intentionally **not perfect**, which is desirable for synthetic data meant to reflect real-world uncertainty.

Global SHAP analysis shows that:
- Premium pressure
- Tenure
- Coverage level
- Risk-related attributes  

are the dominant drivers of lapse risk, matching domain intuition.

---

## 7. Retrieval-Augmented Generation (RAG)

### Architecture
A lightweight RAG pipeline was implemented with:
- Two separate document corpora:
  - **Lapse prevention**
  - **Lead conversion**
- TF-IDF–based retrieval (no heavy dependencies)
- Deterministic, citation-based generation

### Faithfulness
All generated recommendations are strictly grounded in retrieved documents. Each step includes explicit `[Doc#]` citations, enabling direct verification and preventing hallucination.

### Probability-in-Prompt
In the lapse-prevention branch, the **predicted lapse probability** is explicitly injected into the reasoning flow (e.g., “predicted lapse probability = 35.56%”). This demonstrates how ML outputs can be operationalized by downstream decision systems.

---

## 8. What This Assignment Demonstrates

This assignment demonstrates:
- Awareness of **temporal leakage and causality**
- Ability to design **realistic synthetic data**
- Correct use of **time-based evaluation**
- Practical understanding of **churn modeling**
- Integration of **ML predictions with decision support via RAG**
- Emphasis on **faithfulness and explainability**, not just raw accuracy

---

## 9. Limitations & Extensions

- The lapse mechanism is simplified and does not include claims or payment history.
- Future extensions could include rolling behavioral features or survival analysis.
- The RAG generator is rule-based for determinism; an LLM could be introduced while preserving retrieval constraints.

---

## Final Takeaway

The focus of this assignment is not maximizing metrics, but demonstrating **correct modeling discipline**, **temporal reasoning**, and **production-oriented thinking**. All design choices were made to reflect how such a system would be built, evaluated, and audited in a real-world environment.
