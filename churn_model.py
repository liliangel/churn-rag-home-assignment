from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


class ChurnModelTrainer:
    """
    End-to-end churn / lapse model trainer.

    Responsibilities:
      - strict temporal train/val/test split
      - one-hot encoding (region)
      - XGBoost DMatrices
      - Optuna hyperparameter tuning (<= n_trials)
      - metric computation: AUC-PR + precision@1%, precision@5%
      - saving model, metrics.json, and global feature-importance plot
    """

    def __init__(self, out_dir: str = "out", n_trials: int = 30, seed: int = 42):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.n_trials = n_trials
        self.seed = seed

        # Placeholders to attach state during the pipeline
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.X_train_enc = None
        self.X_val_enc = None
        self.X_test_enc = None
        self.feature_names = None

        self.dtrain = None
        self.dval = None
        self.dtest = None

        self.best_model = None
        self.best_params = None
        self.study = None
        self.y_test_proba = None
        self.metrics = None

    # ---------------------------------------------------------
    # 1) Strict temporal split
    # ---------------------------------------------------------
    def strict_time_split(self, df: pd.DataFrame):
        """
        Perform a strict temporal split of the panel dataset into
        train / validation / test sets.

        Split logic:
            - Train = first 8 months
            - Val   = months 9 and 10
            - Test  = remaining months
        """
        TARGET = "lapse_next_3m"

        # Columns that must NOT be used as model features
        drop_cols = [
            TARGET,
            "policy_id",
            "month",
            "lapsed_this_month",
            "is_active",
        ]

        # Deliberate leakage trap columns (must be excluded)
        leak_cols = [c for c in df.columns if c.startswith("post_event_")]

        features = [c for c in df.columns if c not in drop_cols + leak_cols]

        df = df.copy()
        months = sorted(df["month"].unique())

        train_months = months[:8]
        val_months = months[8:10]
        test_months = months[10:]

        train = df[df["month"].isin(train_months)]
        val = df[df["month"].isin(val_months)]
        test = df[df["month"].isin(test_months)]

        self.X_train, self.y_train = train[features], train[TARGET]
        self.X_val, self.y_val = val[features], val[TARGET]
        self.X_test, self.y_test = test[features], test[TARGET]

        # Keep policy_id & month for prediction outputs
        self.test_ids = test["policy_id"].reset_index(drop=True)
        self.test_months = test["month"].reset_index(drop=True)

    # ---------------------------------------------------------
    # 2) Encoding + DMatrices
    # ---------------------------------------------------------
    def encode_and_build_dmatrices(self):
        """
        One-hot encode 'region' consistently across splits,
        align columns, and build XGBoost DMatrices.
        """
        # One-hot encode region column
        self.X_train_enc = pd.get_dummies(
            self.X_train, columns=["region"], drop_first=False
        )
        self.X_val_enc = pd.get_dummies(
            self.X_val, columns=["region"], drop_first=False
        )
        self.X_test_enc = pd.get_dummies(
            self.X_test, columns=["region"], drop_first=False
        )

        # Align columns (ensure same dummy set across splits)
        self.X_train_enc, self.X_val_enc = self.X_train_enc.align(
            self.X_val_enc, join="left", axis=1, fill_value=0
        )
        self.X_train_enc, self.X_test_enc = self.X_train_enc.align(
            self.X_test_enc, join="left", axis=1, fill_value=0
        )

        # Save feature names for later (SHAP / importance plot)
        self.feature_names = list(self.X_train_enc.columns)

        # Build DMatrices
        self.dtrain = xgb.DMatrix(self.X_train_enc, label=self.y_train)
        self.dval = xgb.DMatrix(self.X_val_enc, label=self.y_val)
        self.dtest = xgb.DMatrix(self.X_test_enc, label=self.y_test)

    # ---------------------------------------------------------
    # 3) Optuna tuning + XGBoost training
    # ---------------------------------------------------------
    def tune_with_optuna(self):
        """
        Run a light Optuna hyperparameter search on XGBoost using DMatrices.
        Optimizes AUC-PR on the validation set.
        """
        y_val = self.dval.get_label()

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 10.0
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "lambda": trial.suggest_float("lambda", 0.0, 5.0),
                "alpha": trial.suggest_float("alpha", 0.0, 5.0),
                "seed": self.seed,
            }

            evals = [(self.dtrain, "train"), (self.dval, "val")]

            booster = xgb.train(
                params=params,
                dtrain=self.dtrain,
                num_boost_round=2000,
                evals=evals,
                early_stopping_rounds=30,
                verbose_eval=False,
            )

            y_val_pred = booster.predict(self.dval)
            auc_pr = average_precision_score(y_val, y_val_pred)
            return auc_pr

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(
            objective, n_trials=self.n_trials, show_progress_bar=False
        )

        self.best_params = self.study.best_params
        self.best_params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "seed": self.seed,
            }
        )

        # Train best model
        evals = [(self.dtrain, "train"), (self.dval, "val")]
        self.best_model = xgb.train(
            params=self.best_params,
            dtrain=self.dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        # Predictions on test set
        self.y_test_proba = self.best_model.predict(self.dtest)

    # ---------------------------------------------------------
    # 4) Metrics
    # ---------------------------------------------------------
    @staticmethod
    def compute_metrics_aucpr_precisionk(y_true, y_proba, ks=(0.01, 0.05)):
        """
        Compute AUC-PR and precision@k for given labels and probabilities.
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        auc_pr = average_precision_score(y_true, y_proba)

        order = np.argsort(-y_proba)
        y_true_sorted = y_true[order]

        metrics = {"auc_pr": float(auc_pr)}
        n = len(y_true)

        for k in ks:
            top_k = max(1, int(np.floor(k * n)))
            top_k_true = y_true_sorted[:top_k]
            precision_k = top_k_true.mean() if top_k_true.size > 0 else 0.0
            key = f"precision_at_{int(k * 100)}pct"
            metrics[key] = float(precision_k)

        return metrics

    def compute_metrics(self):
        """
        Compute metrics on the test set and attach best_params and best_val_auc_pr.
        """
        self.metrics = self.compute_metrics_aucpr_precisionk(
            y_true=self.y_test,
            y_proba=self.y_test_proba,
            ks=(0.01, 0.05),
        )
        self.metrics["best_params"] = self.best_params
        self.metrics["best_val_auc_pr"] = float(self.study.best_value)

    # ---------------------------------------------------------
    # 5) Save model, metrics, global importance plot
    # ---------------------------------------------------------
    def save_artifacts(self):
        """
        Save:
          - trained model as model.json
          - metrics.json
          - global importance plot as shap_global_bar.png
            (true SHAP if available, otherwise XGBoost feature importance)
        """
        model_path = self.out_dir / "model.json"
        self.best_model.save_model(str(model_path))

        metrics_path = self.out_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        shap_path = self.out_dir / "shap_global_bar.png"

        # Try SHAP bar plot first
        try:
            explainer = shap.TreeExplainer(self.best_model)
            # Use a sample of the encoded training data for speed
            sample = self.X_train_enc.sample(
                n=min(2000, len(self.X_train_enc)), random_state=self.seed
            )
            shap_values = explainer.shap_values(sample)

            plt.figure(figsize=(8, 6))
            shap.summary_plot(
                shap_values,
                sample,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False,
            )
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150)
            plt.close()
            print("SHAP global bar plot created successfully.")
        except Exception as e:
            print(
                "WARNING: SHAP TreeExplainer failed, falling back to XGBoost feature importance."
            )
            

            importance = self.best_model.get_score(importance_type="gain")
            if not importance:
                importance = self.best_model.get_score(importance_type="weight")

            if not importance:
                print("No feature importance available; skipping plot.")
            else:
                feat_names = np.array(list(importance.keys()))
                scores = np.array(list(importance.values()), dtype=float)
                order = np.argsort(-scores)
                feat_names = feat_names[order]
                scores = scores[order]

                plt.figure(figsize=(8, 6))
                plt.barh(feat_names[::-1], scores[::-1])
                plt.xlabel("Importance")
                plt.title("Global feature importance (XGBoost fallback)")
                plt.tight_layout()
                plt.savefig(shap_path, dpi=150)
                plt.close()
                print("Saved fallback global importance plot to:", shap_path)

        print(f"Saved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved global plot to: {shap_path}")

    #---------------------------------------------------------
    # 6) Infering DTest
    #---------------------------------------------------------
    def save_test_predictions(self, filename: str = "test_predictions.csv"):
        """
        Run inference on the test set and save a CSV with:
          - original test features (X_test)
          - true target (lapse_next_3m)
          - predicted lapse probability (lapse_proba)

        The file is written into self.out_dir / filename.
        """
        if self.y_test_proba is None or self.X_test is None or self.y_test is None:
            raise RuntimeError("Model not trained or test set not available. Call run(df) first.")

        # Build output dataframe from test features
        df_out = self.X_test.copy().reset_index(drop=True)

        #  Add identifiers
        df_out["policy_id"] = self.test_ids
        df_out["month"] = self.test_months

        # Add true label
        df_out["lapse_next_3m_true"] = self.y_test.reset_index(drop=True)

        # Add predicted probability from best_model on dtest
        df_out["lapse_proba"] = self.y_test_proba

        out_path = self.out_dir / filename
        df_out.to_csv(out_path, index=False)

        print(f"Saved test predictions to: {out_path}")
        return out_path

    # ---------------------------------------------------------
    # 7)Public entry point
    # ---------------------------------------------------------
    def run(self, df: pd.DataFrame):
        """
        Full pipeline:
          - temporal split
          - encoding + DMatrices
          - Optuna tuning + training
          - metrics
          - artifact saving

        Returns
        -------
        metrics : dict
        best_params : dict
        preds_path : pathlib.Path
            Path to the saved test_predictions.csv file.
        """
        self.df = df

        # 1) Split
        self.strict_time_split(df)

        # 2) Encode + DMatrices
        self.encode_and_build_dmatrices()

        # 3) Tune + train
        self.tune_with_optuna()

        # 4) Metrics
        self.compute_metrics()

        # 5) Save artifacts
        self.save_artifacts()

        # 6) Save test predictions (inference on dtest)
        preds_path = self.save_test_predictions(filename="test_predictions.csv")

        return self.metrics, self.best_params, preds_path
