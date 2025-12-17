import pandas as pd
from pathlib import Path
from data_generation import generate_insurance_panel
from churn_model import ChurnModelTrainer
from rag import run_rag_from_test_predictions


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    print("1) Generating synthetic insurance panel...")
    df = generate_insurance_panel()
    data_path = out_dir / "data.csv"
    df.to_csv(data_path, index=False)
    print(f"Saved data to: {data_path}")

    print("2) Training churn model with Optuna tuning...")
    trainer = ChurnModelTrainer(out_dir="out", n_trials=30, seed=42)
    metrics, best_params, preds_path = trainer.run(df)

    print("\nDone.")
    print("Test metrics:")
    print(f"  AUC-PR:          {metrics['auc_pr']:.6f}")
    print(f"  Precision@1%:    {metrics.get('precision_at_1pct', float('nan')):.6f}")
    print(f"  Precision@5%:    {metrics.get('precision_at_5pct', float('nan')):.6f}")

    print("\nBest hyperparameters (Optuna):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\nArtifacts written to 'out/':")
    print("  - data.csv")
    print("  - model.json")
    print("  - metrics.json")
    print("  - shap_global_bar.png")
    print(f"  - {preds_path.name}")


    # -----------------------------
    # RAG FLOW (required)
    # -----------------------------
    print("\n3) Running RAG flows (TF-IDF) ...")

    # Load test predictions df produced by churn_model.py
    test_pred_df = pd.read_csv(preds_path)

    # This runs BOTH required branches:
    # - Lapse prevention: pick 3 test customers (high/median/low), use last lapse_proba,
    #   inject probability into reasoning, generate 3-step plan with [Doc#] citations.
    # - Lead conversion: define 3 leads and generate 3-step plans with [Doc#] citations.
    run_rag_from_test_predictions(test_pred_df=test_pred_df, out_dir=out_dir)

    print("\nRAG artifacts written to 'out/':")
    print("  - rag_docs/ (two corpora: lapse + lead markdown)")
    print("  - rag_report.md (human-readable plans with [Doc#] citations)")
    print("  - rag_outputs.json (structured output)")


if __name__ == "__main__":
    main()
