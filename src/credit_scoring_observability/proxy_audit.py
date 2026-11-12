"""Teste de proxy: dá para prever o atributo excluído (sex) a partir das features do modelo?

Se as features do modelo conseguem prever `sex` bem, elas carregam informação sobre o
atributo mesmo sem ele ter sido usado (risco de discriminação por proxy). AUC perto de
0.5 significa que quase não há informação sobre `sex` nas features.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from credit_scoring_observability.prep import REFERENCE_CSV
from credit_scoring_observability.train import feature_columns

PROTECTED = "sex"
REPORT_MD = Path("reports/proxy_audit.md")
REPORT_JSON = Path("reports/proxy_audit.json")


def new_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1
    )


def cv_auc(X: pd.DataFrame, y) -> tuple[float, float]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(new_model(), X, y, cv=folds, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def main() -> None:
    df = pd.read_csv(REFERENCE_CSV)
    features = feature_columns(df)
    X, y = df[features], (df[PROTECTED] == 2).astype(int)

    auc, auc_std = cv_auc(X, y)
    # Controle: com o rótulo embaralhado não há sinal nenhum; o AUC deve ficar perto de 0.5.
    shuffled = pd.Series(
        np.random.default_rng(42).permutation(y.to_numpy()), index=y.index
    )
    chance, chance_std = cv_auc(X, shuffled)

    top = (
        pd.Series(new_model().fit(X, y).feature_importances_, index=features)
        .sort_values(ascending=False)
        .head(5)
    )

    verdict = "negligible" if auc < 0.55 else "weak" if auc < 0.65 else "notable"
    lines = [
        "| Test | AUC (5-fold CV) |",
        "|---|---|",
        f"| Predict `{PROTECTED}` from the model features | {auc:.3f} ± {auc_std:.3f} |",
        f"| Control: shuffled labels (should be ~0.5) | {chance:.3f} ± {chance_std:.3f} |",
        "",
        f"Proxy signal (heuristic scale: <0.55 negligible, 0.55-0.65 weak, >0.65 notable): **{verdict}**.",
        "",
        "Top features for predicting the protected attribute:",
        "",
        "| Feature | Importance |",
        "|---|---|",
        *[f"| {name} | {value:.3f} |" for name, value in top.items()],
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps(
            {
                "protected_attribute": PROTECTED,
                "auc_cv": round(auc, 4),
                "auc_cv_std": round(auc_std, 4),
                "control_auc": round(chance, 4),
                "verdict": verdict,
                "top_features": {k: round(float(v), 4) for k, v in top.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"AUC prevendo '{PROTECTED}' pelas features: {auc:.3f} ± {auc_std:.3f}")
    print(f"Controle (rótulo embaralhado):           {chance:.3f} ± {chance_std:.3f}")
    print(f"Sinal de proxy: {verdict}")
    print("\nFeatures mais usadas:")
    print(top.round(3).to_string())


if __name__ == "__main__":
    main()
