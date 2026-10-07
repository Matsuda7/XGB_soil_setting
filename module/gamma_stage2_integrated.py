"""Step 5: integrated terrain, geology, and SolphyJ second-stage comparison."""
import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from module.gamma_features import load_dem, terrain_at, model_frame
from module.gamma_stage2 import exclude_heldout, regenerate
from module.gamma_stage2_geology import attach_geology
from module.paths import project_path
from module.validation import split_data
from module.xgb_common import build_regressor, regression_metrics


SOLPHYJ_COLUMNS = ["representative_total_sand", "representative_silt", "representative_clay"]
TERRAIN_COLUMNS = ["slope", "curvature"]


def build_missing_preprocessor(numeric, categorical):
    transformers = [("numeric", "passthrough", list(numeric))]
    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical)))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def fit_evaluation_missing(parts, numeric, categorical, parameters):
    train, validation, test = parts
    preprocessor = build_missing_preprocessor(numeric, categorical)
    xtrain = preprocessor.fit_transform(model_frame(train, numeric, categorical))
    xvalidation = preprocessor.transform(model_frame(validation, numeric, categorical))
    model = build_regressor(parameters)
    model.fit(xtrain, train.target, eval_set=[(xvalidation, validation.target)], verbose=False)
    predicted = model.predict(preprocessor.transform(model_frame(test, numeric, categorical)))
    return model, preprocessor, predicted


def attach_solphyj(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    solphy = pd.read_csv(project_path(config["solphyj_features"]))
    cols = ["record_id", *SOLPHYJ_COLUMNS]
    result = frame.merge(solphy[cols], on="record_id", how="left", validate="one_to_one")
    result["solphyj_available"] = result[SOLPHYJ_COLUMNS].notna().all(axis=1).astype(int)
    return result


def attach_terrain(frame: pd.DataFrame, stage1: dict, out) -> tuple[pd.DataFrame, dict]:
    terrain = terrain_at(frame, load_dem(stage1["config"]), stage1["config"])
    result = frame.copy()
    for col in TERRAIN_COLUMNS:
        result[col] = terrain[col].to_numpy()
    if not np.isfinite(result[TERRAIN_COLUMNS]).all(axis=None):
        raise ValueError("Nonfinite terrain features; cannot preserve Step 5 population")
    metadata = {
        "dem": stage1["config"]["spatial"]["dem_input"],
        "crs": stage1["config"]["spatial"]["grid_crs"],
        "resolution_m": stage1["config"]["spatial"]["grid_spacing_m"],
        "slope_unit": "degree",
        "curvature_unit": "1/m",
        "algorithm": "Same module.gamma_features.terrain_at implementation used in Step 2.",
        "missing_policy": "Same as Step 2: negative/nonfinite DEM cells -> 0; outside offset samples -> 0; outside center -> NaN, abort if nonfinite derived features remain.",
        "affected_rows": int(terrain.dem_missing.sum()),
        "affected_measured": int(terrain.loc[frame.label_source == "measured", "dem_missing"].sum()),
        "affected_pseudo": int(terrain.loc[frame.label_source == "predicted", "dem_missing"].sum()),
        "missing_counts": terrain[TERRAIN_COLUMNS].isna().sum().to_dict(),
    }
    result[["record_id", "x", "y", *TERRAIN_COLUMNS]].assign(dem_missing=terrain.dem_missing).to_csv(
        out / "terrain_features.csv", index=False
    )
    return result, metadata


def solphyj_split_audit(seed, train, val, test, pseudo, safe) -> pd.DataFrame:
    rows = []
    for split, data in [
        ("train_measured", train),
        ("validation_measured", val),
        ("test_measured", test),
        ("safe_pseudo_candidates", safe),
        ("regenerated_pseudo", pseudo),
    ]:
        rows.append(
            {
                "seed": seed,
                "split": split,
                "count": len(data),
                "solphyj_available": int(data.solphyj_available.sum()),
                "solphyj_missing": int((data.solphyj_available == 0).sum()),
                "solphyj_missing_rate": float((data.solphyj_available == 0).mean()) if len(data) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def make_f4_predictions(test, predicted, condition) -> pd.DataFrame:
    result = test[
        [
            "record_id",
            "boring_id",
            "x",
            "y",
            "elevation",
            "slope",
            "curvature",
            "geology",
            *SOLPHYJ_COLUMNS,
            "solphyj_available",
            "target",
        ]
    ].copy()
    result = result.rename(columns={"target": "observed_gamma"})
    result["predicted_gamma"] = predicted
    result["residual"] = result["observed_gamma"] - result["predicted_gamma"]
    result["training_data"] = condition
    return result[
        [
            "training_data",
            "record_id",
            "boring_id",
            "observed_gamma",
            "predicted_gamma",
            "residual",
            "x",
            "y",
            "elevation",
            "slope",
            "curvature",
            "geology",
            *SOLPHYJ_COLUMNS,
            "solphyj_available",
        ]
    ]


def delta_table(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("F1-F0", "F1", "F0"),
        ("F2-F0", "F2", "F0"),
        ("F3-F0", "F3", "F0"),
        ("F4-F0", "F4", "F0"),
        ("F4-F1", "F4", "F1"),
        ("F4-F2", "F4", "F2"),
        ("F4-F3", "F4", "F3"),
    ]
    rows = []
    index = scores.set_index(["seed", "condition", "model"])
    for seed in sorted(scores.seed.unique()):
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


def check_reproducibility(scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    checks = []
    references = [
        ("F0", "Step1_baseline", project_path(config["baseline_output"]) / "metrics.csv", {"measured_only": "measured_only", "D_first_augmented": "D_first_augmented"}),
        ("F1", "Step2_M3", project_path(config["terrain_output"]) / "metrics.csv", {"measured_only": "measured_only", "D_first_augmented": "D_first_augmented"}),
        ("F2", "Step3_G1", project_path(config["geology_output"]) / "metrics.csv", {"measured_only": "measured_only", "D_first_augmented": "D_first_augmented"}),
    ]
    current = scores.set_index(["seed", "condition", "model"])
    for model, label, path, condition_map in references:
        ref = pd.read_csv(path)
        for seed in config["split_seeds"]:
            for condition, ref_condition in condition_map.items():
                if label == "Step1_baseline":
                    match = ref.loc[(ref["mode"] == "borehole") & (ref.seed == seed) & (ref.model == ref_condition)].iloc[0]
                elif label == "Step2_M3":
                    match = ref.loc[(ref.seed == seed) & (ref.model == "M3") & (ref.condition == ref_condition)].iloc[0]
                else:
                    match = ref.loc[(ref.seed == seed) & (ref.model == "G1") & (ref.condition == ref_condition)].iloc[0]
                cur = current.loc[(seed, condition, model)]
                row = {"seed": seed, "condition": condition, "model": model, "reference": label}
                for metric in ["r2", "rmse", "mae"]:
                    row[f"delta_{metric}"] = float(cur[metric] - match[metric])
                checks.append(row)
    return pd.DataFrame(checks)


def write_readme(out, summary, delta_summary, stability, split_counts, solphyj_audit, reproducibility, terrain_metadata):
    lines = [
        "# Step 5：地形・地質・SolphyJ統合評価",
        "",
        "実測γ280件を共通評価母集団とし、SolphyJ欠損では行を除外していません。代表Sand/Silt/ClayがないSSGはNaNのままXGBoostへ渡しました。数値特徴量には中央値補完を入れず、geologyのみStep 3と同じOneHotEncoder(handle_unknown=ignore)でtrainingデータだけからエンコードしています。",
        "最終モデル決定、全域γ分布作成、ハイパーパラメータ調整、疑似γ重み変更、SolphyJ欠損補間、Bulk density追加、PCA/log-ratio変換、特徴量選択は実行していません。",
        "",
        "## 使用特徴量",
        "",
        "|モデル|特徴量|",
        "|---|---|",
        "|F0|x, y, elevation|",
        "|F1|x, y, elevation, slope, curvature|",
        "|F2|x, y, elevation, geology|",
        "|F3|x, y, elevation, representative_total_sand, representative_silt, representative_clay|",
        "|F4|x, y, elevation, slope, curvature, geology, representative_total_sand, representative_silt, representative_clay|",
        "",
        "SolphyJの3変数はSSG代表の表層土壌特性であり、ボーリング深度の実粒度ではありません。Sand+Silt+ClayはStep 4-3でほぼ100%と確認済みで、強い組成制約がありますが、今回は3変数のまま比較しました。",
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
        "## 差分",
        "",
        "R²は正、RMSE・MAEは負が改善です。",
        "",
        "|training data|comparison|ΔR²|ΔRMSE|ΔMAE|",
        "|---|---|---:|---:|---:|",
    ]
    for (condition, comparison), row in delta_summary.iterrows():
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
        "|seed|train measured|validation|test|safe pseudo|regenerated pseudo|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in split_counts.itertuples(index=False):
        lines.append(f"|{row.seed}|{row.train_measured}|{row.validation_count}|{row.test_count}|{row.safe_pseudo_candidates}|{row.regenerated_pseudo}|")
    lines += [
        "",
        "## SolphyJ欠損監査",
        "",
        "実測γ280件のうちSolphyJ 3物性あり225件、なし55件。疑似γ815件のうちあり587件、なし228件。欠損を理由に行削除していません。",
        "",
        "|seed|split|count|SolphyJあり|SolphyJなし|欠損率|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in solphyj_audit.itertuples(index=False):
        lines.append(f"|{row.seed}|{row.split}|{row.count}|{row.solphyj_available}|{row.solphyj_missing}|{row.solphyj_missing_rate:.1%}|")
    max_abs = reproducibility[[c for c in reproducibility.columns if c.startswith("delta_")]].abs().max()
    lines += [
        "",
        "## 既存Stepとの整合性",
        "",
        f"F0はStep 1 baseline、F1はStep 2 M3、F2はStep 3 G1と比較しました。最大絶対差はR²={max_abs['delta_r2']:.3g}、RMSE={max_abs['delta_rmse']:.3g}、MAE={max_abs['delta_mae']:.3g}です。詳細は`reproducibility_check.csv`に保存しました。",
        "",
        "## leakage audit",
        "",
        "各seedでvalidation/test実測γをStage 1教師モデルの学習・疑似γ生成から除外しました。heldout borehole overlap=0を各`borehole_* / leakage_audit.json`に保存しています。",
        "",
        "## 地形・地質処理",
        "",
        f"slope/curvatureはStep 2と同じ`terrain_at`で計算しました。DEM={terrain_metadata['dem']}、CRS={terrain_metadata['crs']}、解像度={terrain_metadata['resolution_m']} m、slope単位=degree、curvature単位=1/m。欠損処理もStep 2と同じです。",
        "geologyはStep 3と同じsymbolを使用し、未知カテゴリはone-hot全ゼロになります。",
        "",
        "## 保存ファイル",
        "",
        "- `metrics.csv`：各seed・各モデルのR²/RMSE/MAE",
        "- `summary.csv`：5 seed平均とSD",
        "- `deltas.csv`、`delta_summary.csv`：指定差分",
        "- `stability.csv`：改善seed数",
        "- `split_counts.csv`：train/validation/test件数",
        "- `solphyj_missing_audit.csv`：SolphyJあり/なし件数",
        "- `unseen_categories.csv`：geology未知カテゴリ監査",
        "- `reproducibility_check.csv`：既存Stepとの差",
        "- `borehole_*/F4_*_test_predictions.csv`：F4誤差分析用CSV",
        "",
        "再実行：`python -m module.gamma_stage2_integrated`",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gamma/stage2_integrated.json")
    args = parser.parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    if config["sample_weight"] != 1.0:
        raise ValueError("Step 5 expects equal unit weights")
    out = project_path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(project_path(config["input"]))
    stage1 = json.loads(project_path(config["stage1_settings"]).read_text(encoding="utf-8"))
    frame = attach_solphyj(frame, config)
    frame, terrain_metadata = attach_terrain(frame, stage1, out)
    frame = attach_geology(frame, out)

    measured = frame.loc[frame.label_source == "measured"].copy()
    candidates = frame.loc[frame.label_source == "predicted"].copy()
    if len(measured) != 280 or len(candidates) != 815:
        raise ValueError(f"Unexpected rows: measured={len(measured)}, candidates={len(candidates)}")
    required_nonfinite = ["x", "y", "elevation", *TERRAIN_COLUMNS, "target"]
    if not np.isfinite(measured[required_nonfinite].to_numpy(float)).all():
        raise ValueError("Nonfinite required measured values outside SolphyJ features")
    if not np.isfinite(candidates[["x", "y", "elevation", *TERRAIN_COLUMNS]].to_numpy(float)).all():
        raise ValueError("Nonfinite required pseudo candidate values outside SolphyJ features")

    rows = []
    split_rows = []
    solphyj_audits = []
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
        solphyj_audits.append(solphyj_split_audit(seed, train, val, test, pseudo, safe))
        (dest / "leakage_audit.json").write_text(
            json.dumps(
                {
                    "measured_population": 280,
                    "pseudo_candidate_population": 815,
                    "test_labels_used_in_teacher": False,
                    "validation_labels_used_in_teacher": False,
                    "reused_saved_pseudo_targets": False,
                    "same_weight": True,
                    "heldout_borehole_overlap": 0,
                    "solphyj_missing_rows_removed": 0,
                },
                indent=2,
            )
            + "\n"
        )
        for model_name, model_config in config["models"].items():
            numeric = model_config["numeric"]
            categorical = model_config["categorical"]
            for condition, learning in [("measured_only", train), ("D_first_augmented", augmented)]:
                model, _, prediction = fit_evaluation_missing((learning, val, test), numeric, categorical, config["xgb_parameters"])
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
                if model_name == "F4":
                    make_f4_predictions(test, prediction, condition).to_csv(
                        dest / f"F4_{condition}_test_predictions.csv", index=False
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
        print(f"Completed Step 5 seed={seed}", flush=True)

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
    solphyj_audit = pd.concat(solphyj_audits, ignore_index=True)
    solphyj_audit.to_csv(out / "solphyj_missing_audit.csv", index=False)
    if category_rows:
        pd.DataFrame(category_rows).to_csv(out / "unseen_categories.csv", index=False)
    reproducibility = check_reproducibility(scores, config)
    reproducibility.to_csv(out / "reproducibility_check.csv", index=False)
    (out / "settings.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / "terrain_metadata.json").write_text(json.dumps(terrain_metadata, indent=2) + "\n", encoding="utf-8")
    write_readme(out, summary, delta_summary, stability, split_counts, solphyj_audit, reproducibility, terrain_metadata)
    print(summary.to_string())
    print(delta_summary.to_string())
    print(stability.to_string(index=False))
    print(reproducibility[[c for c in reproducibility.columns if c.startswith("delta_")]].abs().max().to_string())


if __name__ == "__main__":
    main()
