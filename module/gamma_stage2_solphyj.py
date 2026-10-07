"""Step 4-3: evaluate SolphyJ representative properties in second-stage gamma models."""
import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from module.gamma_stage2 import exclude_heldout, regenerate
from module.gamma_stage2_geology import attach_geology
from module.gamma_training import fit_evaluation
from module.paths import project_path
from module.validation import split_data
from module.xgb_common import regression_metrics


SOLPHYJ_COLUMNS = ["representative_total_sand", "representative_silt", "representative_clay"]


def load_feature_frame(config: dict, out) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(project_path(config["input"]))
    solphy = pd.read_csv(project_path(config["solphyj_features"]))
    cols = ["record_id", *SOLPHYJ_COLUMNS]
    frame = frame.merge(solphy[cols], on="record_id", how="left", validate="one_to_one")
    frame = attach_geology(frame, out)
    frame["has_solphyj_three"] = frame[SOLPHYJ_COLUMNS].notna().all(axis=1)
    composition = frame.loc[
        (frame.label_source == "measured") & frame.has_solphyj_three, SOLPHYJ_COLUMNS
    ].sum(axis=1)
    stats = {
        "measured_common_rows": int(composition.size),
        "mean": float(composition.mean()),
        "std": float(composition.std(ddof=1)),
        "min": float(composition.min()),
        "max": float(composition.max()),
    }
    return frame, stats


def write_readme(out, summary, deltas, stability, split_counts, composition_stats, unavailable) -> None:
    lines = [
        "# Step 4-3：SolphyJ代表土壌物性の第二段階評価",
        "",
        "最終モデル決定、全域γ分布作成、Bulk density追加、slope/curvature追加、疑似γ重み変更、欠損補間、ハイパーパラメータ調整は実行していません。",
        "主比較はSolphyJのSand/Silt/Clayを3つとも取得できる実測γ225件に限定し、S0〜S3すべて同じ評価母集団から孔単位分割を作成しました。疑似γ候補も3物性が揃う587件に限定しました。",
        "",
        "## SolphyJ組成チェック",
        "",
        "|対象|mean|std|min|max|",
        "|---|---:|---:|---:|---:|",
        f"|Sand+Silt+Clay合計（実測225件）|{composition_stats['mean']:.4f}|{composition_stats['std']:.4f}|{composition_stats['min']:.4f}|{composition_stats['max']:.4f}|",
        "",
        "3変数は組成データとして強く依存しますが、今回は指定どおり3変数のまま試験しました。値はボーリング深度の実粒度ではなく、SSG代表の表層土壌特性です。",
        "",
        "## 5 seed平均",
        "",
        "|training data|model|R²|RMSE|MAE|",
        "|---|---|---:|---:|---:|",
    ]
    for (condition, model), row in summary.iterrows():
        lines.append(
            f"|{condition}|{model}|"
            f"{row[('r2', 'mean')]:.4f} ± {row[('r2', 'std')]:.4f}|"
            f"{row[('rmse', 'mean')]:.4f} ± {row[('rmse', 'std')]:.4f}|"
            f"{row[('mae', 'mean')]:.4f} ± {row[('mae', 'std')]:.4f}|"
        )
    lines += [
        "",
        "RMSE・MAEの単位はkN/m³。SDは5 seedの標本標準偏差です。",
        "",
        "## 差分",
        "",
        "R²は正、RMSE・MAEは負が改善です。",
        "",
        "|training data|comparison|ΔR²|ΔRMSE|ΔMAE|",
        "|---|---|---:|---:|---:|",
    ]
    for (condition, comparison), row in deltas.groupby(["condition", "comparison"])[["delta_r2", "delta_rmse", "delta_mae"]].agg(["mean", "std"]).iterrows():
        lines.append(
            f"|{condition}|{comparison}|"
            f"{row[('delta_r2', 'mean')]:+.4f} ± {row[('delta_r2', 'std')]:.4f}|"
            f"{row[('delta_rmse', 'mean')]:+.4f} ± {row[('delta_rmse', 'std')]:.4f}|"
            f"{row[('delta_mae', 'mean')]:+.4f} ± {row[('delta_mae', 'std')]:.4f}|"
        )
    lines += [
        "",
        "## 改善seed数",
        "",
        "|training data|comparison|R²改善|RMSE改善|MAE改善|",
        "|---|---|---:|---:|---:|",
    ]
    for row in stability.itertuples(index=False):
        lines.append(f"|{row.condition}|{row.comparison}|{row.r2_improved}|{row.rmse_improved}|{row.mae_improved}|")
    lines += [
        "",
        "## 分割件数",
        "",
        "|seed|train measured|validation|test|pseudo candidates after leakage exclusion|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in split_counts.itertuples(index=False):
        lines.append(f"|{row.seed}|{row.train_measured}|{row.validation_count}|{row.test_count}|{row.safe_pseudo_candidates}|")
    lines += [
        "",
        "## 欠損・未知カテゴリ",
        "",
        "SolphyJ特徴量が揃わない実測55件と疑似228件は今回の母集団から除外しました。除外IDは`excluded_solphyj_records.csv`に保存しました。",
        "geologyはStep 3と同じsymbolをカテゴリ変数として使用し、OneHotEncoder(handle_unknown=ignore)により各seed・各条件のtrainingデータだけでエンコードします。test側の未知カテゴリは地質one-hot部分が全ゼロになります。",
        "",
        "## 保存ファイル",
        "",
        "- `metrics.csv`：seed別のR²/RMSE/MAEと件数",
        "- `summary.csv`：5 seed平均とSD",
        "- `deltas.csv`：S1-S0、S2-S0、S3-S2、S3-S1のseed別差分",
        "- `delta_summary.csv`：差分の平均とSD",
        "- `stability.csv`：改善したseed数",
        "- `split_counts.csv`：各seedのtrain/validation/test件数",
        "- `composition_stats.json`：Sand+Silt+Clay合計の統計",
        "- `excluded_solphyj_records.csv`：今回除外したSolphyJ欠損レコード",
        "- `geology_features.csv`、`category_counts.csv`、`unseen_categories.csv`：地質カテゴリ監査",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def delta_table(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = [("S1-S0", "S1", "S0"), ("S2-S0", "S2", "S0"), ("S3-S2", "S3", "S2"), ("S3-S1", "S3", "S1")]
    rows = []
    index = scores.set_index(["seed", "condition", "model"])
    for seed in scores.seed.unique():
        for condition in scores.condition.unique():
            for label, after, before in pairs:
                a = index.loc[(seed, condition, after)]
                b = index.loc[(seed, condition, before)]
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "comparison": label,
                        "delta_r2": a.r2 - b.r2,
                        "delta_rmse": a.rmse - b.rmse,
                        "delta_mae": a.mae - b.mae,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gamma/stage2_solphyj.json")
    args = parser.parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    if config["sample_weight"] != 1.0:
        raise ValueError("Step 4-3 expects equal unit weights")
    out = project_path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    frame, composition_stats = load_feature_frame(config, out)
    stage1 = json.loads(project_path(config["stage1_settings"]).read_text(encoding="utf-8"))

    measured_all = frame.loc[frame.label_source == "measured"].copy()
    candidates_all = frame.loc[frame.label_source == "predicted"].copy()
    measured = measured_all.loc[measured_all.has_solphyj_three].copy()
    candidates = candidates_all.loc[candidates_all.has_solphyj_three].copy()
    excluded = frame.loc[~frame.has_solphyj_three, ["record_id", "label_source", "boring_id", "x", "y", "elevation"]].copy()
    excluded.to_csv(out / "excluded_solphyj_records.csv", index=False)

    if len(measured) != 225 or len(candidates) != 587:
        raise ValueError(f"Unexpected SolphyJ-complete counts: measured={len(measured)}, pseudo={len(candidates)}")
    required = sorted({feature for model in config["models"].values() for feature in model["numeric"]})
    if not np.isfinite(measured[required + ["target"]].to_numpy(float)).all():
        raise ValueError("Nonfinite measured numeric feature or target")
    if not np.isfinite(candidates[required].to_numpy(float)).all():
        raise ValueError("Nonfinite pseudo numeric feature")

    rows = []
    split_rows = []
    category_rows = []
    for seed in config["split_seeds"]:
        dest = out / f"borehole_{seed}"
        dest.mkdir(exist_ok=True)
        validation = {
            "mode": "borehole",
            "test_size": config["test_size"],
            "validation_size": config["validation_size"],
            "random_state": seed,
        }
        train, val, test = split_data(measured, validation, target_column="target")
        heldout = pd.concat([val, test])
        safe = exclude_heldout(candidates, heldout)
        pseudo = regenerate(train, safe, stage1["features"], stage1["config"]["xgb_parameters"], config, seed, dest)
        augmented = pd.concat([train, pseudo], ignore_index=True)
        assert not set(augmented.boring_id) & set(heldout.boring_id)

        pd.concat(
            [
                part[["record_id", "boring_id", "_validation_group"]].assign(split=name)
                for name, part in zip(("train", "validation", "test"), (train, val, test))
            ]
        ).to_csv(dest / "measured_split.csv", index=False)
        pseudo.to_csv(dest / "regenerated_pseudo.csv", index=False)
        augmented.to_csv(dest / "training.csv", index=False)
        split_rows.append(
            {
                "seed": seed,
                "train_measured": len(train),
                "validation_count": len(val),
                "test_count": len(test),
                "safe_pseudo_candidates": len(safe),
                "regenerated_pseudo": len(pseudo),
            }
        )
        (dest / "leakage_audit.json").write_text(
            json.dumps(
                {
                    "common_measured_population": 225,
                    "common_pseudo_candidate_population": 587,
                    "test_labels_used_in_teacher": False,
                    "validation_labels_used_in_teacher": False,
                    "reused_saved_pseudo_targets": False,
                    "same_weight": True,
                    "heldout_borehole_overlap": 0,
                },
                indent=2,
            )
            + "\n"
        )
        for model_name, model_config in config["models"].items():
            numeric = model_config["numeric"]
            categorical = model_config["categorical"]
            for condition, learning in [("measured_only", train), ("D_first_augmented", augmented)]:
                model, _, prediction = fit_evaluation((learning, val, test), numeric, categorical, config["xgb_parameters"])
                metrics = regression_metrics(test.target, prediction)
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "model": model_name,
                        "train_measured": len(train),
                        "train_pseudo": len(learning) - len(train),
                        "train_count": len(learning),
                        "validation_count": len(val),
                        "test_count": len(test),
                        "best_n_estimators": model.best_iteration + 1,
                        **metrics,
                    }
                )
                for column in categorical:
                    known = set(learning[column])
                    category_rows.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "model": model_name,
                            "feature": column,
                            "train_categories": len(known),
                            "unseen_validation": int((~val[column].isin(known)).sum()),
                            "unseen_test": int((~test[column].isin(known)).sum()),
                            "unseen_test_categories": ";".join(sorted(set(test[column]) - known)),
                        }
                    )
                test[["record_id", "boring_id", "target"]].assign(prediction=prediction).to_csv(
                    dest / f"{model_name}_{condition}_predictions.csv", index=False
                )
        print(f"Completed Step 4-3 seed={seed}", flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(out / "metrics.csv", index=False)
    summary = scores.groupby(["condition", "model"])[["r2", "rmse", "mae"]].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    deltas = delta_table(scores)
    deltas.to_csv(out / "deltas.csv", index=False)
    delta_summary = deltas.groupby(["condition", "comparison"])[["delta_r2", "delta_rmse", "delta_mae"]].agg(["mean", "std"])
    delta_summary.to_csv(out / "delta_summary.csv")
    stability = deltas.groupby(["condition", "comparison"]).agg(
        r2_improved=("delta_r2", lambda s: int((s > 0).sum())),
        rmse_improved=("delta_rmse", lambda s: int((s < 0).sum())),
        mae_improved=("delta_mae", lambda s: int((s < 0).sum())),
    ).reset_index()
    stability.to_csv(out / "stability.csv", index=False)
    split_counts = pd.DataFrame(split_rows)
    split_counts.to_csv(out / "split_counts.csv", index=False)
    if category_rows:
        pd.DataFrame(category_rows).to_csv(out / "unseen_categories.csv", index=False)
    (out / "composition_stats.json").write_text(json.dumps(composition_stats, indent=2) + "\n")
    (out / "settings.json").write_text(json.dumps(config, indent=2) + "\n")
    write_readme(out, summary, deltas, stability, split_counts, composition_stats, excluded)
    print(summary.to_string())
    print(delta_summary.to_string())
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
