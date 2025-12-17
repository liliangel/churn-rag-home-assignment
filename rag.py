from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# 1) RAG docs corpora (markdown)
# ---------------------------

LAPSE_DOCS = [
    ("Doc1_grace_period.md", """# Grace Period & Non-Payment Basics
- Send a short reminder with a due date and a single action link.
- Confirm what happens during the grace window and the deadline.
- If no response after two attempts, escalate to agent outreach.
"""),
    ("Doc2_agent_outreach.md", """# Agent Outreach Playbook
- Use agent outreach for high-risk customers and complex needs.
- Keep it lightweight: 1 call + 1 follow-up message.
- Structure: acknowledge → offer help → confirm next step + deadline.
"""),
    ("Doc3_payment_plans.md", """# Payment Plans & Billing Flexibility
- Offer installments (2–4) or align billing date to paycheck timing.
- Present max 2 options to avoid confusion.
- Confirm total cost and schedule clearly.
"""),
    ("Doc4_loyalty_discounts.md", """# Loyalty Discounts & Retention Offers
- Use targeted, time-limited loyalty offers for long-tenure customers.
- Avoid blanket discounts; communicate duration and conditions.
- Consider non-price bundle benefits when possible.
"""),
    ("Doc5_seasonality.md", """# Seasonality & Timing
- Mid-year budget pressure and holidays can increase missed billing.
- Use concise messages and proactive reminders during high-stress periods.
- Early-week outreach often gets faster responses.
"""),
    ("Doc6_smoker_coaching.md", """# Smoker Coaching & Value Reframing
- Avoid judgment; reframe value as continuity and stability.
- Provide one supportive resource and one financial option if price-sensitive.
- If they engage, schedule a follow-up touchpoint.
"""),
]

LEAD_DOCS = [
    ("Doc1_messaging_by_segment.md", """# Messaging by Segment
- Price-sensitive: emphasize transparency and flexible options.
- Family-focused: emphasize dependents protection and stability.
- High-coverage: emphasize protection and support.
- Use one primary benefit and one clear CTA.
"""),
    ("Doc2_touchpoint_cadence.md", """# Touchpoint Cadence
- Day 0: confirmation + next step
- Day 2: reminder + one value point
- Day 5–7: objection handling + final CTA
- Match channel: chat→email+SMS, referral→call window
"""),
    ("Doc3_objection_handling.md", """# Objection Handling
- “Too expensive”: offer smaller starting option and explain tradeoffs.
- “Need to think”: propose a scheduled follow-up.
- “Don’t trust”: emphasize clarity, terms, and support.
- Handle one objection at a time; end with one next action.
"""),
    ("Doc4_value_props.md", """# Value Propositions
- Protection (coverage stability)
- Convenience (easy pay, simple renewal)
- Support (agent help)
- Choose 1–2 pillars; avoid feature dumping.
"""),
    ("Doc5_trial_discount_guidelines.md", """# Trial/Discount Guidelines
- Use discounts only for price-sensitive leads with clear intent.
- Keep offers time-limited and clearly defined.
- Reinforce value, not just price.
"""),
    ("Doc6_channel_playbooks.md", """# Channel Playbooks
- Web form: fast response + quote link + short benefit.
- Referral: credibility + call scheduling.
- Social: low friction, simple CTA, trust tone.
"""),
]


def write_corpus(base_dir: Path) -> None:
    (base_dir / "lapse").mkdir(parents=True, exist_ok=True)
    (base_dir / "lead").mkdir(parents=True, exist_ok=True)
    for fname, txt in LAPSE_DOCS:
        (base_dir / "lapse" / fname).write_text(txt.strip() + "\n", encoding="utf-8")
    for fname, txt in LEAD_DOCS:
        (base_dir / "lead" / fname).write_text(txt.strip() + "\n", encoding="utf-8")


# ---------------------------
# 2) TF-IDF Retriever
# ---------------------------

@dataclass
class Doc:
    doc_id: int
    name: str
    text: str


class TfidfRetriever:
    def __init__(self, docs: List[Doc]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Doc, float]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.doc_matrix).ravel()
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(self.docs[i], float(sims[i])) for i in idxs]


def load_docs_from_dir(dir_path: Path) -> List[Doc]:
    md_files = sorted(dir_path.glob("Doc*.md"))
    docs: List[Doc] = []
    for i, fp in enumerate(md_files, start=1):
        docs.append(Doc(doc_id=i, name=fp.name, text=fp.read_text(encoding="utf-8")))
    return docs


# ---------------------------
# 3) Faithful plan generation (grounded in retrieved docs)
# ---------------------------

def _doc_has(doc_text: str, keywords: List[str]) -> bool:
    t = doc_text.lower()
    return any(k.lower() in t for k in keywords)


def generate_lapse_plan(prob: float, customer_row: Dict[str, Any], retrieved: List[Doc]) -> str:
    # Probability must clearly appear inside the reasoning flow
    if prob >= 0.70:
        band = "HIGH"
    elif prob >= 0.35:
        band = "MEDIUM"
    else:
        band = "LOW"

    # Select steps strictly from retrieved docs
    steps: List[str] = []
    for d in retrieved:
        if _doc_has(d.text, ["grace", "reminder", "deadline"]) and len(steps) < 1:
            steps.append(f"Send a concise grace-period reminder with deadline and one action link. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["agent", "call", "follow-up"]) and len(steps) < 2:
            steps.append(f"Do lightweight agent outreach (1 call + 1 follow-up) with one clear next step. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["installments", "billing", "payment"]) and len(steps) < 3:
            steps.append(f"Offer payment flexibility (max 2 options): installments or billing-date alignment. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["loyalty", "discount"]) and len(steps) < 3 and band != "LOW":
            steps.append(f"If appropriate, use a targeted time-limited loyalty offer with clear conditions. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["seasonality", "holidays", "timing"]) and len(steps) < 3:
            steps.append(f"Time outreach proactively during higher-stress periods; keep messaging short. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["smoker", "value", "resource"]) and len(steps) < 3 and int(customer_row.get("is_smoker", 0)) == 1:
            steps.append(f"Reframe value for smoker segment and provide one supportive resource plus one financial option. [Doc{d.doc_id}]")

    # Ensure exactly 3 steps using only retrieved docs
    if len(steps) < 3:
        for d in retrieved:
            if len(steps) >= 3:
                break
            steps.append(f"Apply the retention guidance from retrieved sources and confirm a deadline. [Doc{d.doc_id}]")
    steps = steps[:3]

    # Reasoning must include probability explicitly
    return (
        f"Customer (policy_id={customer_row['policy_id']}, month={customer_row['month']}): "
        f"predicted lapse probability = **{prob:.2%}** → risk band = **{band}**.\n"
        f"Grounded in the retrieved retention sources, 3-step plan:\n"
        f"1) {steps[0]}\n"
        f"2) {steps[1]}\n"
        f"3) {steps[2]}"
    )


def generate_lead_plan(lead: Dict[str, Any], retrieved: List[Doc]) -> str:
    steps: List[str] = []
    for d in retrieved:
        if _doc_has(d.text, ["segment", "cta"]) and len(steps) < 1:
            steps.append(f"Use one primary benefit matched to segment (**{lead['segment']}**) with one clear CTA. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["day 0", "day 2", "day 5", "cadence"]) and len(steps) < 2:
            steps.append(f"Apply a simple cadence for **{lead['channel']}** leads (Day 0, Day 2, Day 5–7). [Doc{d.doc_id}]")
        if _doc_has(d.text, ["objection", "expensive", "trust", "think"]) and len(steps) < 3:
            steps.append(f"Handle the main objection (**{lead['objection']}**) with one response and one next action. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["protection", "convenience", "support"]) and len(steps) < 3:
            steps.append(f"Reinforce 1–2 value pillars relevant to needs (**{lead['needs']}**). [Doc{d.doc_id}]")
        if _doc_has(d.text, ["discount"]) and len(steps) < 3 and "expensive" in lead["objection"].lower():
            steps.append(f"If needed, use a time-limited, clearly defined discount and keep value-led messaging. [Doc{d.doc_id}]")
        if _doc_has(d.text, ["web form", "referral", "social", "channel"]) and len(steps) < 3:
            steps.append(f"Adapt message tone and CTA to the channel (**{lead['channel']}**) and reduce friction. [Doc{d.doc_id}]")

    if len(steps) < 3:
        for d in retrieved:
            if len(steps) >= 3:
                break
            steps.append(f"Use retrieved lead playbook guidance and end with a single CTA. [Doc{d.doc_id}]")
    steps = steps[:3]

    return (
        f"Lead {lead['lead_id']} (age={lead['age']}, region={lead['region']}, channel={lead['channel']}): "
        f"needs={lead['needs']}, objection={lead['objection']}\n"
        f"Grounded in retrieved lead sources, 3-step plan:\n"
        f"1) {steps[0]}\n"
        f"2) {steps[1]}\n"
        f"3) {steps[2]}"
    )


# ---------------------------
# 4) Selecting demo customers from test_pred_df
# ---------------------------

def last_prediction_per_policy(test_pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the last row per policy_id (latest m_idx or latest month) and keep its lapse_proba.
    """
    df = test_pred_df.copy()

    # Ensure proper ordering
    if "m_idx" in df.columns:
        df = df.sort_values(["policy_id", "m_idx"])
    else:
        df = df.sort_values(["policy_id", "month"])

    last = df.groupby("policy_id", as_index=False).tail(1).reset_index(drop=True)
    return last


def pick_high_median_low(last_df: pd.DataFrame) -> pd.DataFrame:
    """
    last_df must include policy_id, month, lapse_proba.
    """
    df = last_df.sort_values("lapse_proba").reset_index(drop=True)
    low = df.iloc[[0]]
    mid = df.iloc[[len(df) // 2]]
    high = df.iloc[[-1]]
    return pd.concat([high, mid, low], ignore_index=True)


# ---------------------------
# 5) Main run function
# ---------------------------

def run_rag_from_test_predictions(
    test_pred_df: pd.DataFrame,
    out_dir: Path = Path("out"),
    top_k: int = 3
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Write docs
    docs_dir = out_dir / "rag_docs"
    write_corpus(docs_dir)

    # 2) Load docs + build retrievers
    lapse_docs = load_docs_from_dir(docs_dir / "lapse")
    lead_docs = load_docs_from_dir(docs_dir / "lead")
    lapse_retriever = TfidfRetriever(lapse_docs)
    lead_retriever = TfidfRetriever(lead_docs)

    # 3) Pick demo customers: last prediction per policy, then high/median/low
    last_df = last_prediction_per_policy(test_pred_df)
    demo_df = pick_high_median_low(last_df)

    lapse_outputs = []
    for _, r in demo_df.iterrows():
        customer = r.to_dict()
        prob = float(customer["lapse_proba"])

        query = (
            f"retention plan grace period payment plan agent outreach "
            f"risk={prob:.2f} smoker={customer.get('is_smoker','')} "
            f"agent={customer.get('has_agent','')} tenure={customer.get('tenure_m','')} "
            f"premium={customer.get('premium','')} region={customer.get('region','')}"
        )
        retrieved = [d for (d, _s) in lapse_retriever.search(query, top_k=top_k)]
        plan = generate_lapse_plan(prob, customer, retrieved)

        lapse_outputs.append({
            "policy_id": int(customer["policy_id"]),
            "month": str(customer["month"]),
            "lapse_probability": prob,
            "retrieved_docs": [{"doc_id": d.doc_id, "name": d.name} for d in retrieved],
            "plan": plan
        })

    # 4) Lead conversion: define 3 leads
    lead_profiles = [
        {
            "lead_id": "lead_1",
            "age": 28,
            "region": "WEST",
            "channel": "social",
            "segment": "price-sensitive",
            "needs": "basic coverage quickly",
            "objection": "Too expensive"
        },
        {
            "lead_id": "lead_2",
            "age": 41,
            "region": "SOUTH",
            "channel": "web form",
            "segment": "family-focused",
            "needs": "dependents protection and stability",
            "objection": "Need to think"
        },
        {
            "lead_id": "lead_3",
            "age": 55,
            "region": "EAST",
            "channel": "referral",
            "segment": "high-coverage needs",
            "needs": "higher coverage and support",
            "objection": "Don’t trust insurance"
        },
    ]

    lead_outputs = []
    for lp in lead_profiles:
        query = f"{lp['segment']} {lp['channel']} objection {lp['objection']} value proposition cadence"
        retrieved = [d for (d, _s) in lead_retriever.search(query, top_k=top_k)]
        plan = generate_lead_plan(lp, retrieved)
        lead_outputs.append({
            "lead_id": lp["lead_id"],
            "profile": lp,
            "retrieved_docs": [{"doc_id": d.doc_id, "name": d.name} for d in retrieved],
            "plan": plan
        })

    result = {
        "lapse_prevention": lapse_outputs,
        "lead_conversion": lead_outputs
    }

    # Save JSON
    (out_dir / "rag_outputs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Save markdown report
    lines = ["# RAG Report\n"]

    lines.append("## Lapse prevention (3 demo customers: high / median / low)\n")
    for item in lapse_outputs:
        lines.append(f"### Policy {item['policy_id']} @ {item['month']}\n")
        lines.append("Retrieved: " + ", ".join([f"[Doc{d['doc_id']}] {d['name']}" for d in item["retrieved_docs"]]) + "\n")
        lines.append(item["plan"] + "\n")

    lines.append("\n## Lead conversion (3 leads)\n")
    for item in lead_outputs:
        lines.append(f"### {item['lead_id']}\n")
        lines.append("Retrieved: " + ", ".join([f"[Doc{d['doc_id']}] {d['name']}" for d in item["retrieved_docs"]]) + "\n")
        lines.append(item["plan"] + "\n")

    (out_dir / "rag_report.md").write_text("\n".join(lines), encoding="utf-8")

    return result


if __name__ == "__main__":
    print("Import run_rag_from_test_predictions(test_pred_df) and call it from run.py.")
