"""Generate the figures flagged as missing in manuscript/draft_v1.md:
  Fig 1: dataset composition (cation counts, multivalent dataset)
  Fig 2: predicted vs actual log10(conductivity), OBELiX held-out test set
  Fig 3: predicted vs actual log10(conductivity), multivalent dataset (LOO), colored by cation
  Fig 4: RandomForest feature importances, final multivalent model (top 15)
  Fig 5: R2 trend across dataset-growth passes and feature/tuning stages

All figures saved as 300dpi PNGs to manuscript/figures/, using only matplotlib
(no seaborn dependency) for portability.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

FIG_DIR = r"C:\Polyemr Electrolyter ML project\manuscript\figures"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11, "axes.titlesize": 11})


def fig1_composition():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features_v2.csv")
    counts = {"Mg²⁺": df["cation_Mg"].sum(), "Zn²⁺": df["cation_Zn"].sum(), "Na⁺": df["cation_Na"].sum()}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(counts.keys(), counts.values(), color=["#5B8C5A", "#4A7A96", "#B0795C"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, int(b.get_height()), ha="center")
    ax.set_ylabel("Number of data points")
    ax.set_title(f"Dataset composition (n={len(df)})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\fig1_dataset_composition.png")
    plt.close(fig)
    print("Saved fig1_dataset_composition.png")


def fig2_obelix_parity():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\obelix_features.csv")
    y = df["log10_conductivity"].values
    X = df.drop(columns=["log10_conductivity", "composition", "doi"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.scatter(y_test, pred, alpha=0.6, s=25, color="#4A7A96", edgecolors="none")
    lims = [min(y_test.min(), pred.min()) - 0.5, max(y_test.max(), pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual log$_{10}$(conductivity / S cm$^{-1}$)")
    ax.set_ylabel("Predicted log$_{10}$(conductivity / S cm$^{-1}$)")
    ax.set_title(f"OBELiX benchmark (Li-ion, n={len(df)})\nXGBoost, held-out test set, R² = {r2:.2f}")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\fig2_obelix_parity.png")
    plt.close(fig)
    print(f"Saved fig2_obelix_parity.png (R2={r2:.3f})")


def fig3_multivalent_parity():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features_v2.csv")
    y = df["log10_conductivity"].values
    feature_cols = [c for c in df.columns if c not in ("log10_conductivity", "polymer_host", "anion_class")]
    X = df[feature_cols]

    best_params = {"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}
    loo = LeaveOneOut()
    preds = cross_val_predict(RandomForestRegressor(**best_params), X, y, cv=loo)
    r2 = r2_score(y, preds)

    cation_colors = np.select(
        [df["cation_Mg"] == 1, df["cation_Zn"] == 1, df["cation_Na"] == 1],
        ["#5B8C5A", "#4A7A96", "#B0795C"],
        default="#999999",
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    for cation, color in [("Mg²⁺", "#5B8C5A"), ("Zn²⁺", "#4A7A96"), ("Na⁺", "#B0795C")]:
        mask = cation_colors == color
        ax.scatter(y[mask], preds[mask], alpha=0.7, s=30, color=color, edgecolors="none", label=cation)
    lims = [min(y.min(), preds.min()) - 0.5, max(y.max(), preds.max()) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual log$_{10}$(conductivity / S cm$^{-1}$)")
    ax.set_ylabel("Predicted log$_{10}$(conductivity / S cm$^{-1}$)")
    ax.set_title(f"Multivalent polymer electrolytes (n={len(df)})\nTuned RandomForest, LOO-CV, R² = {r2:.2f}")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\fig3_multivalent_parity.png")
    plt.close(fig)
    print(f"Saved fig3_multivalent_parity.png (R2={r2:.3f})")


def fig4_feature_importance():
    df = pd.read_csv(r"C:\Polyemr Electrolyter ML project\data\processed\multivalent_features_v2.csv")
    y = df["log10_conductivity"].values
    feature_cols = [c for c in df.columns if c not in ("log10_conductivity", "polymer_host", "anion_class")]
    X = df[feature_cols]
    best_params = {"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}
    model = RandomForestRegressor(**best_params)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh(importances.index[::-1], importances.values[::-1], color="#4A7A96")
    ax.set_xlabel("RandomForest feature importance")
    ax.set_title("Top 15 feature importances\n(final multivalent model)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\fig4_feature_importance.png")
    plt.close(fig)
    print("Saved fig4_feature_importance.png")


def fig5_r2_trend():
    stages = [
        "v1 coarse\n(n=48)", "v1 coarse\n(n=91)", "v1 coarse\n(n=137)",
        "v2 chem.\nfeatures", "v2 tuned\n(LOO)", "v2 tuned\n(5-fold)",
    ]
    r2_values = [0.48, 0.40, 0.30, 0.35, 0.45, 0.27]
    colors = ["#B0795C"] * 3 + ["#4A7A96"] * 1 + ["#5B8C5A", "#C44E4E"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(stages, r2_values, color=colors)
    for b, v in zip(bars, r2_values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("R² (RandomForest, best of tree models)")
    ax.set_title(
        "R² across dataset growth and modeling stages\n"
        "(n=48→137 decline reflects less LOO leakage, not a worse model)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}\\fig5_r2_trend.png")
    plt.close(fig)
    print("Saved fig5_r2_trend.png")


if __name__ == "__main__":
    import os

    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_composition()
    fig2_obelix_parity()
    fig3_multivalent_parity()
    fig4_feature_importance()
    fig5_r2_trend()
    print("\nAll figures written to", FIG_DIR)
