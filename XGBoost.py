from __future__ import annotations
import numpy as np
import pandas as pd
import argparse
import pathlib
import warnings
from typing import Tuple, List, Dict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

THIS_DIR = pathlib.Path(__file__).resolve().parent


def LoadData(
    dataPath: str | pathlib.Path = THIS_DIR / "data.csv",
    labelsPath: str | pathlib.Path = THIS_DIR / "labels.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(dataPath, index_col=0)
    y = pd.read_csv(labelsPath, index_col=0).squeeze("columns")
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "prosrtate cancer": "prostate cancer",
            "colon cancer": "colon cancer", 
            "breast cancer": "breast cancer",
            "lung cancer": "lung cancer",
        })
        .str.title()  
    )
    return X, y


def PreprocessCounts(X: pd.DataFrame) -> pd.DataFrame:
    """Scale each sample to relative abundances (rows sum to 1)."""
    return X.div(X.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

def BinaryMetrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec

def EvaluateOneRest(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: str,
    *,
    n_splits: int,
    random_state: int,
) -> Tuple[float, float, float, float]:
    y_bin = (y == positive_label).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    sens_rf: List[float] = []
    spec_rf: List[float] = []
    sens_xgb: List[float] = []
    spec_xgb: List[float] = []

    for train_idx, test_idx in skf.split(X, y_bin):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]

  # XGBoost
        xgb = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,           
            n_jobs=-1,
            random_state=random_state,
        )
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        sens, spec = BinaryMetrics(y_test, rf.predict(X_test))
        sens_rf.append(sens)
        spec_rf.append(spec)

      
        xgb.fit(X_train, y_train)
        sens, spec = BinaryMetrics(y_test, xgb.predict(X_test))
        sens_xgb.append(sens)
        spec_xgb.append(spec)

    return (
        float(np.mean(sens_rf)),
        float(np.mean(spec_rf)),
        float(np.mean(sens_xgb)),
        float(np.mean(spec_xgb)),
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    X, y = LoadData()
    X = PreprocessCounts(X)

    results: List[Dict[str, float]] = []

    print(f"=== {args.folds}-fold CV  —  Sensitivity / Specificity ===\n")
    header = "Cancer            RF_sens     RF_spec    | XGB_sens / spec"
    print(header)
    print("-" * len(header))

    for cancer in y.unique():
        sens_rf, spec_rf, sens_xgb, spec_xgb = EvaluateOneRest(
            X, y, positive_label=cancer, n_splits=args.folds, random_state=args.seed
        )
        print(f"{cancer:<15}  {sens_rf:6.3f}     {spec_rf:6.3f}  |  {sens_xgb:6.3f} / {spec_xgb:6.3f}")
        results.append(
            {
                "Cancer": cancer,
                "RF_sens": sens_rf,
                "RF_spec": spec_rf,
                "XGB_sens": sens_xgb,
                "XGB_spec": spec_xgb,
            }
        )

    out_path = THIS_DIR / "cv_metrics.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nMetrics table saved ➜ {out_path.relative_to(THIS_DIR)}")

if __name__ == "__main__":
    main()
