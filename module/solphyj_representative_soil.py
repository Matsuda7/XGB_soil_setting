"""Step 4-2: build SSG representative SolphyJ soil properties; no model training."""
import json
from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from module.paths import ROOT


AUDIT = ROOT / "gamma_test/soil_inventory_audit"
OUT = ROOT / "gamma_test/solphyj_representative_soil"
TARGET_PROPERTIES = {
    "Total sand": "total_sand",
    "Silt": "silt",
    "Clay": "clay",
}


def valid_property_rows(solphy: pd.DataFrame, prop: str) -> pd.DataFrame:
    rows = solphy.loc[
        (solphy["property"] == prop)
        & solphy["Mean"].notna()
        & (solphy["Number of samples"] > 0)
    ].copy()
    rows["weighted_sum"] = rows["Mean"] * rows["Number of samples"]
    return rows


def representative_property(solphy: pd.DataFrame, prop: str, short: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = valid_property_rows(solphy, prop)
    grouped = rows.groupby("SSG code", dropna=False)
    summary = grouped.agg(
        soil_series_group=("Soil Series Group", "first"),
        records_used=("Mean", "size"),
        total_samples=("Number of samples", "sum"),
        min_mean=("Mean", "min"),
        max_mean=("Mean", "max"),
        weighted_sum=("weighted_sum", "sum"),
    ).reset_index()
    summary[f"representative_{short}"] = summary["weighted_sum"] / summary["total_samples"]
    summary = summary.drop(columns="weighted_sum")
    renamed = summary.rename(
        columns={
            "records_used": f"{short}_records_used",
            "total_samples": f"{short}_total_samples",
            "min_mean": f"{short}_min_mean",
            "max_mean": f"{short}_max_mean",
            "soil_series_group": "Soil Series Group",
        }
    )
    audit = renamed[
        [
            "SSG code",
            "Soil Series Group",
            f"{short}_records_used",
            f"{short}_total_samples",
            f"{short}_min_mean",
            f"{short}_max_mean",
            f"representative_{short}",
        ]
    ].copy()
    audit = audit.rename(
        columns={
            f"{short}_records_used": "records_used",
            f"{short}_total_samples": "total_samples",
            f"{short}_min_mean": "min_mean",
            f"{short}_max_mean": "max_mean",
            f"representative_{short}": "weighted_mean",
        }
    )
    return renamed, audit


def thickness_table(solphy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = valid_property_rows(solphy, "Thickness")
    rows["Horizon"] = pd.to_numeric(rows["Horizon"], errors="coerce").astype("Int64")
    rows = rows.dropna(subset=["Horizon"])
    grouped = rows.groupby(["SSG code", "Soil Series Group", "Horizon"], dropna=False).agg(
        thickness_records_used=("Mean", "size"),
        thickness_total_samples=("Number of samples", "sum"),
        thickness_min_mean=("Mean", "min"),
        thickness_max_mean=("Mean", "max"),
        weighted_sum=("weighted_sum", "sum"),
    ).reset_index()
    grouped["mean_thickness"] = grouped["weighted_sum"] / grouped["thickness_total_samples"]
    grouped = grouped.drop(columns="weighted_sum")
    grouped = grouped.sort_values(["SSG code", "Horizon"])
    grouped["cumulative_mean_thickness"] = grouped.groupby("SSG code")["mean_thickness"].cumsum()

    wide_parts = []
    for value in ["mean_thickness", "cumulative_mean_thickness"]:
        wide = grouped.pivot(index="SSG code", columns="Horizon", values=value)
        wide.columns = [f"H{int(c)}_{value}" for c in wide.columns]
        wide_parts.append(wide)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    wide["cumulative_mean_thickness"] = grouped.groupby("SSG code")["mean_thickness"].sum().reindex(wide["SSG code"]).to_numpy()
    return wide, grouped


def build_representative_table(solphy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    dictionary = pd.read_csv(AUDIT / "ssg_dictionary.csv")
    dictionary = dictionary.rename(columns={"包括土壌統群コード": "SSG code", "包括土壌統群名": "Soil Series Group"})
    representative = dictionary[["SSG code", "Soil Series Group"]].drop_duplicates("SSG code").copy()
    audits = {}
    for prop, short in TARGET_PROPERTIES.items():
        table, audit = representative_property(solphy, prop, short)
        audits[short] = audit
        cols = [
            "SSG code",
            f"representative_{short}",
            f"{short}_records_used",
            f"{short}_total_samples",
            f"{short}_min_mean",
            f"{short}_max_mean",
        ]
        representative = representative.merge(table[cols], on="SSG code", how="left")
    thickness_wide, thickness_audit = thickness_table(solphy)
    representative = representative.merge(thickness_wide, on="SSG code", how="left")
    return representative, audits, thickness_audit


def load_soil_map() -> gpd.GeoDataFrame:
    folder = ROOT / "data/raw/soil_inventory"
    zips = [
        p for p in folder.rglob("*.zip")
        if any(name.lower().endswith(".shp") for name in zipfile.ZipFile(p).namelist())
    ]
    if len(zips) != 1:
        raise ValueError("Expected one soil-map archive under data/raw/soil_inventory")
    return gpd.read_file("zip://" + str(zips[0]), encoding="cp932")


def prop_mask_code_counts(soil: gpd.GeoDataFrame, representative: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    spatial = json.loads((ROOT / "config/spatial.json").read_text(encoding="utf-8"))
    projected = soil.to_crs(spatial["grid_crs"]).copy()
    projected["geometry"] = projected.geometry.make_valid()
    projected["SSerGrCD"] = projected["SSerGrCD"].astype("string")
    bounds = spatial["property_mask_bounds"]
    spacing = float(spatial["property_mask_spacing_m"])

    prop_codes = {
        short: set(representative.loc[representative[f"representative_{short}"].notna(), "SSG code"].astype(str))
        for short in TARGET_PROPERTIES.values()
    }
    all_three_codes = set.intersection(*prop_codes.values())
    map_codes = set(projected["SSerGrCD"].dropna().astype(str))
    missing_codes = sorted(map_codes - all_three_codes)
    missing_geoms = {}
    for code in missing_codes:
        geom = shapely.union_all(projected.loc[projected["SSerGrCD"] == code, "geometry"].to_numpy())
        shapely.prepare(geom)
        missing_geoms[code] = geom
    covered_geom = shapely.union_all(projected.geometry.to_numpy())
    shapely.prepare(covered_geom)

    totals = {code: 0 for code in missing_codes}
    total_mask_points = 0
    covered_points = 0
    chunk_x: list[np.ndarray] = []
    chunk_y: list[np.ndarray] = []
    chunk_rows = 500

    def flush_chunk() -> None:
        nonlocal covered_points
        if not chunk_x:
            return
        x = np.concatenate(chunk_x)
        y = np.concatenate(chunk_y)
        chunk_x.clear()
        chunk_y.clear()
        covered_points += int(np.count_nonzero(shapely.intersects_xy(covered_geom, x, y)))
        for code, geom in missing_geoms.items():
            totals[code] += int(np.count_nonzero(shapely.intersects_xy(geom, x, y)))

    with (ROOT / spatial["property_mask"]).open() as stream:
        for row, line in enumerate(stream):
            mask = np.fromstring(line, sep=" ")
            indices = np.flatnonzero(mask != 0)
            if len(indices) == 0:
                continue
            total_mask_points += len(indices)
            chunk_x.append(bounds["xmin"] + indices * spacing)
            chunk_y.append(np.full(len(indices), bounds["ymin"] + row * spacing))
            if len(chunk_x) >= chunk_rows:
                flush_chunk()
    flush_chunk()

    counts = pd.DataFrame({"SSG code": list(totals.keys()), "prop_mask_points": list(totals.values())})
    if counts.empty:
        counts = pd.DataFrame(columns=["SSG code", "prop_mask_points"])
    counts = counts.merge(representative, on="SSG code", how="left")
    for short in TARGET_PROPERTIES.values():
        counts[f"has_{short}"] = counts[f"representative_{short}"].notna()
    all_cols = [f"has_{short}" for short in TARGET_PROPERTIES.values()]
    counts["has_all_three"] = counts[all_cols].all(axis=1)
    counts["prop_mask_fraction"] = counts["prop_mask_points"] / total_mask_points
    counts = counts.sort_values("prop_mask_points", ascending=False)
    availability = {
        "total_mask_points": int(total_mask_points),
        "assigned_soil_points": int(covered_points),
        "outside_soil_map_points": int(total_mask_points - covered_points),
        "spacing_m": spacing,
        "approx_total_mask_area_km2": float(total_mask_points * spacing * spacing / 1e6),
    }
    for short in TARGET_PROPERTIES.values():
        missing_for_property = map_codes - prop_codes[short]
        unavailable_points = (total_mask_points - covered_points) + sum(totals.get(code, 0) for code in missing_for_property)
        availability[f"{short}_available_points"] = int(total_mask_points - unavailable_points)
        availability[f"{short}_available_fraction"] = float(availability[f"{short}_available_points"] / total_mask_points)
    all_three_unavailable = (total_mask_points - covered_points) + sum(totals.values())
    availability["all_three_available_points"] = int(total_mask_points - all_three_unavailable)
    availability["all_three_available_fraction"] = float(availability["all_three_available_points"] / total_mask_points)
    return counts, availability


def attach_to_gamma(representative: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = pd.read_csv(AUDIT / "teacher_soil_join.csv")
    cols = ["SSG code"] + [f"representative_{short}" for short in TARGET_PROPERTIES.values()]
    attached = joined.merge(representative[cols], left_on="SSerGrCD", right_on="SSG code", how="left")
    for short in TARGET_PROPERTIES.values():
        attached[f"has_{short}"] = attached[f"representative_{short}"].notna()
    attached["has_all_three"] = attached[[f"has_{short}" for short in TARGET_PROPERTIES.values()]].all(axis=1)
    summary = []
    for source, group in attached.groupby("label_source"):
        row = {"label_source": source, "total": len(group)}
        for short in TARGET_PROPERTIES.values():
            row[f"{short}_available"] = int(group[f"has_{short}"].sum())
            row[f"{short}_available_fraction"] = float(group[f"has_{short}"].mean())
        row["all_three_available"] = int(group["has_all_three"].sum())
        row["all_three_available_fraction"] = float(group["has_all_three"].mean())
        summary.append(row)
    return attached, pd.DataFrame(summary)


def write_report(summary: dict, representative: pd.DataFrame, prop_counts: pd.DataFrame, gamma_summary: pd.DataFrame) -> None:
    unavailable = prop_counts.loc[~prop_counts["has_all_three"], ["SSG code", "prop_mask_points", "prop_mask_fraction"]].copy()
    lines = [
        "# Step 4-2：SolphyJ SSG代表土壌物性",
        "",
        "XGBoost学習、性能比較、特徴量重要度、最終γ分布作成は実行していません。Step 4-1の抽出済みSolphyJ原尺度データと土壌図結合結果を使用しました。",
        "",
        "## 使用シートと計算方法",
        "",
        "使用したSolphyJシートはTotal sand、Silt、Clay、Thicknessです。SSG codeは土壌図のSSerGrCDと完全一致で結合し、名称の曖昧一致は使用していません。",
        "Total sand / Silt / Clayは、同一SSG code内のLand use×Horizonレコードについて、Meanが非欠損かつNumber of samples>0の行だけを使い、Σ(Mean×Number of samples)/Σ(Number of samples)で代表値を作成しました。3物性はそれぞれ独立に有効サンプル数を使っています。",
        "ThicknessはSSG×HorizonごとにMean ThicknessをNumber of samplesで加重平均し、Horizon順に累積しました。これは参考情報であり、ボーリング深度へ直接対応させていません。",
        "",
        "## 作成結果",
        "",
        f"SolphyJ辞書のSSG数：{summary['representative']['ssg_total']}。Sand/Silt/Clayの3物性すべての代表値を作成できたSSG数：{summary['representative']['all_three_ssg']}。",
        "",
        "|物性|代表値作成SSG数|合計Number of samples|",
        "|---|---:|---:|",
    ]
    for short, label in [("total_sand", "Total sand"), ("silt", "Silt"), ("clay", "Clay")]:
        lines.append(f"|{label}|{summary['representative'][short + '_ssg']}|{summary['representative'][short + '_total_samples']:.0f}|")
    lines += [
        "",
        "SSGごとの代表値・元サンプル数・最小Mean・最大Mean・Thickness累積値は`ssg_representative_properties.csv`に保存しました。物性別の監査表は`representative_property_audit.csv`、ThicknessのHorizon別監査表は`thickness_by_horizon.csv`です。",
        "",
        "## propマスク対応率",
        "",
        "|項目|取得点数|取得率|",
        "|---|---:|---:|",
    ]
    prop = summary["prop_mask"]
    for short, label in [("total_sand", "representative_total_sand"), ("silt", "representative_silt"), ("clay", "representative_clay")]:
        lines.append(f"|{label}|{prop[short + '_available_points']}|{prop[short + '_available_fraction']:.2%}|")
    lines.append(f"|3物性すべて|{prop['all_three_available_points']}|{prop['all_three_available_fraction']:.2%}|")
    lines += [
        "",
        f"propマスク総点数：{prop['total_mask_points']}、近似面積：{prop['approx_total_mask_area_km2']:.2f} km²。土壌図に割り当てできた点数：{prop['assigned_soil_points']}。",
        "",
        "3物性すべてを取得できなかったSSGは`prop_mask_ssg_property_coverage.csv`で確認できます。主な未取得カテゴリは次のとおり。",
        "",
        "|SSG code|propマスク点数|割合|",
        "|---|---:|---:|",
    ]
    for row in unavailable.head(30).itertuples(index=False):
        lines.append(f"|{row._0}|{int(row.prop_mask_points)}|{row.prop_mask_fraction:.2%}|")
    if len(unavailable) > 30:
        lines.append(f"|...|残り{len(unavailable)-30}分類| |")
    lines += [
        "",
        "## γデータへの付与",
        "",
        "|区分|総数|Sand取得|Silt取得|Clay取得|3物性すべて|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gamma_summary.itertuples(index=False):
        label = "実測γ" if row.label_source == "measured" else "D-priority疑似γ候補"
        lines.append(
            f"|{label}|{row.total}|{row.total_sand_available}|{row.silt_available}|{row.clay_available}|{row.all_three_available}|"
        )
    lines += [
        "",
        "同じx,yに複数深度のγデータがある場合でも、同じSSGなら同じ代表Sand/Silt/Clayを付与しています。今回の値は深度別粒度ではなく、地点が属するSSGの代表的な表層土壌特性です。",
        "",
        "## 欠損処理",
        "",
        "Mean欠損行、Number of samples=0の行は、その物性の代表値計算から除外しました。Sand/Silt/Clayは独立に計算しているため、物性ごとに使用レコード数と合計サンプル数が異なります。対応できない土壌図コード、土壌図外、境界上で取得できない点は代表物性欠損として保持しています。",
        "",
        "## 保存ファイル",
        "",
        "- `ssg_representative_properties.csv`：SSGごとの代表Sand/Silt/Clay、元サンプル数、Thickness累積値",
        "- `representative_property_audit.csv`：物性別の使用レコード数、合計サンプル数、最小Mean、最大Mean、加重平均",
        "- `thickness_by_horizon.csv`：SSG×Horizonの代表Thicknessと累積Thickness",
        "- `prop_mask_ssg_property_coverage.csv`：propマスク内SSGごとの取得可否と点数",
        "- `gamma_teacher_with_solphyj_representatives.csv`：実測γ・疑似γ候補への代表物性付与結果",
        "- `gamma_teacher_solphyj_coverage.csv`：γデータ区分別の取得件数",
        "- `summary.json`：主要集計値",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    solphy_path = AUDIT / "solphy_original_scale.csv"
    if not solphy_path.exists():
        raise FileNotFoundError("Run python -m module.soil_inventory_audit first")
    solphy = pd.read_csv(solphy_path)
    representative, audits, thickness_audit = build_representative_table(solphy)
    representative.to_csv(OUT / "ssg_representative_properties.csv", index=False)
    pd.concat(
        [df.assign(property=label) for label, df in audits.items()],
        ignore_index=True,
    ).to_csv(OUT / "representative_property_audit.csv", index=False)
    thickness_audit.to_csv(OUT / "thickness_by_horizon.csv", index=False)

    soil = load_soil_map()
    prop_counts, prop_summary = prop_mask_code_counts(soil, representative)
    prop_counts.to_csv(OUT / "prop_mask_ssg_property_coverage.csv", index=False)
    gamma_attached, gamma_summary = attach_to_gamma(representative)
    gamma_attached.to_csv(OUT / "gamma_teacher_with_solphyj_representatives.csv", index=False)
    gamma_summary.to_csv(OUT / "gamma_teacher_solphyj_coverage.csv", index=False)

    rep_summary = {"ssg_total": int(len(representative))}
    for short in TARGET_PROPERTIES.values():
        rep_summary[f"{short}_ssg"] = int(representative[f"representative_{short}"].notna().sum())
        rep_summary[f"{short}_total_samples"] = float(representative[f"{short}_total_samples"].sum())
    rep_summary["all_three_ssg"] = int(
        representative[[f"representative_{short}" for short in TARGET_PROPERTIES.values()]].notna().all(axis=1).sum()
    )
    summary = {
        "representative": rep_summary,
        "prop_mask": prop_summary,
        "gamma_teacher": gamma_summary.to_dict(orient="records"),
        "outputs": {
            "directory": str(OUT),
            "representative": "ssg_representative_properties.csv",
            "property_audit": "representative_property_audit.csv",
            "thickness": "thickness_by_horizon.csv",
            "prop_mask": "prop_mask_ssg_property_coverage.csv",
            "gamma_teacher": "gamma_teacher_with_solphyj_representatives.csv",
            "gamma_coverage": "gamma_teacher_solphyj_coverage.csv",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, representative, prop_counts, gamma_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
