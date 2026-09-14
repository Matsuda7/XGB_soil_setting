#!/usr/bin/env python3
"""Extract strength-test results from KuniJiban XML files.

The exact KuniJiban XML schema can differ by provider and delivery year. This
module therefore uses local-name matching and conservative heuristics. Unknown
and ambiguous elements are reported in logs for later schema-specific tuning.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from module.paths import RAW, INTERIM, LOGS


DEFAULT_INPUT_DIR = RAW / "soiltest"
DEFAULT_OUTPUT_DIR = INTERIM / "strength"
DEFAULT_LOG_DIR = LOGS

OUTPUT_COLUMNS = [
    "boring_id", "xml_file", "x", "y", "latitude", "longitude", "surface_z",
    "sample_id", "sample_no", "depth_top", "depth_bottom", "depth_mid", "sample_z",
    "soil_name", "soil_code", "test_type", "test_condition_code", "test_condition",
    "wet_density", "dry_density", "particle_density", "water_content", "void_ratio",
    "degree_of_saturation", "gravel_fraction", "sand_fraction", "silt_fraction",
    "clay_fraction", "maximum_particle_size", "d10", "d50", "uniformity_coefficient",
    "liquid_limit", "plastic_limit", "plasticity_index",
    "shear_strength_total", "phi_total",
    "shear_strength_effective", "phi_effective", "qu", "c_total", "c_effective",
    "unit_shear_strength", "unit_c", "unit_phi", "unit_qu",
]

# These are intentionally aliases rather than a single assumed schema.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "boring_id": ("boringid", "boring_id", "孔番号", "ボーリング番号", "ボーリング名", "ファイル名"),
    "x": ("x", "座標x", "坑口座標x", "平面直角x"),
    "y": ("y", "座標y", "坑口座標y", "平面直角y"),
    "latitude": ("latitude", "緯度"),
    "longitude": ("longitude", "経度"),
    "surface_z": ("surfacez", "孔口標高", "坑口標高", "標高"),
    "sample_id": ("sampleid", "sample_id", "試料id", "試料番号", "試料no", "試料ナンバー"),
    "sample_no": ("sampleno", "sample_no", "試料連番", "試料番号連番"),
    "depth_top": ("depthtop", "試料採取開始深度", "開始深度", "上端深度"),
    "depth_bottom": ("depthbottom", "試料採取終了深度", "終了深度", "下端深度"),
    "depth_mid": ("depthmid", "試料代表深度", "代表深度", "試料深度"),
    "sample_z": ("samplez", "試料採取標高", "試料標高"),
    "soil_name": ("soilname", "土質名", "土質分類名", "地盤材料の分類名", "分類", "土質"),
    "soil_code": ("soilcode", "地盤材料の分類記号", "分類記号", "土質コード"),
    "test_type": ("testtype", "試験名", "試験種別", "強度試験", "試験条件", "せん断試験条件", "せん断試験条件コード"),
    "test_condition_code": ("せん断試験条件コード", "試験条件コード"),
    "test_condition": ("せん断試験条件", "試験条件"),
    "wet_density": ("湿潤密度",),
    "dry_density": ("乾燥密度",),
    "particle_density": ("土粒子密度",),
    "water_content": ("自然含水比",),
    "void_ratio": ("間隙比",),
    "degree_of_saturation": ("飽和度",),
    "gravel_fraction": ("礫分",),
    "sand_fraction": ("砂分",),
    "silt_fraction": ("シルト分",),
    "clay_fraction": ("粘土分",),
    "maximum_particle_size": ("最大粒径",),
    "d10": ("D10",),
    "d50": ("D50",),
    "uniformity_coefficient": ("均等係数",),
    "liquid_limit": ("液性限界",),
    "plastic_limit": ("塑性限界",),
    "plasticity_index": ("塑性指数",),
    "c_total": ("ctotal", "全応力粘着力", "粘着力全応力", "c全応力", "c"),
    "shear_strength_total": ("せん断強さ_全応力", "全応力せん断強さ"),
    "phi_total": ("phitotal", "全応力内部摩擦角", "内部摩擦角全応力", "φ全応力", "phi全応力", "せん断抵抗角_全応力"),
    "c_effective": ("ceffective", "有効応力粘着力", "粘着力有効応力", "c有効応力"),
    "shear_strength_effective": ("せん断強さ_有効応力", "有効応力せん断強さ"),
    "phi_effective": ("phieffective", "有効応力内部摩擦角", "内部摩擦角有効応力", "φ有効応力", "phi有効応力", "せん断抵抗角_有効応力"),
    "qu": ("qu", "一軸圧縮強さ", "一軸圧縮強度", "圧縮強さ"),
}

TEST_ALIASES = {
    "UU": ("uu", "非排水三軸", "非圧密非排水"),
    "CUb": ("cub", "cu b"),
    "CU": ("cu", "圧密非排水"),
    "CD": ("cd", "圧密排水"),
    "一軸圧縮試験": ("一軸圧縮", "一軸圧縮試験", "unconfined", "uniaxial"),
}

TEST_CODE_ALIASES = {"B0521": "UU", "B0522": "CU", "B0523": "CUb", "B0524": "CD"}

STRENGTH_HINTS = (
    "強度", "せん断", "三軸", "一軸", "粘着", "摩擦", "uu", "cu", "cub", "cd", "qu",
)

LOGGER = logging.getLogger(__name__)
UNKNOWN_TAGS: Counter[str] = Counter()


@dataclass
class FileResult:
    """Results and status for one XML file."""

    rows: list[dict[str, object]]
    error: str | None = None


def local_name(tag: str) -> str:
    """Return an XML tag name without namespace or prefix."""
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]


def normalized(value: str) -> str:
    """Normalize names for tolerant alias matching."""
    return re.sub(r"[\s_\-:/（）()。、．.]+", "", value).casefold()


def text_of(element: ET.Element) -> str:
    """Return normalized visible text from an element subtree."""
    return " ".join(part.strip() for part in element.itertext() if part and part.strip())


def safe_number(value: str | None) -> float | str | None:
    """Convert a numeric string safely, preserving non-numeric values as text."""
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.lower() in {"nan", "na", "n/a", "-", "欠測", "未測定"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return cleaned
    try:
        number = float(match.group())
        return int(number) if number.is_integer() else number
    except ValueError:
        return cleaned


def alias_matches(name: str, aliases: Iterable[str]) -> bool:
    """Check whether a tag name matches one of the normalized aliases."""
    candidate = normalized(name)
    return any(normalized(alias) == candidate for alias in aliases)


def descendant_values(element: ET.Element) -> dict[str, str]:
    """Collect simple leaf values and attributes from an element subtree."""
    values: dict[str, str] = {}
    for child in element.iter():
        name = local_name(child.tag)
        value = (child.text or "").strip()
        if value:
            values.setdefault(normalized(name), value)
        for attr_name, attr_value in child.attrib.items():
            if attr_value.strip():
                values.setdefault(normalized(local_name(attr_name)), attr_value.strip())
    return values


def find_value(values: dict[str, str], field: str) -> str | None:
    """Find a field value using its alias list."""
    for alias in FIELD_ALIASES[field]:
        value = values.get(normalized(alias))
        if value is not None:
            return value
    return None


def test_type_from_text(value: str) -> str:
    """Map a test description to a stable output category."""
    code = value.strip().upper()
    if code in TEST_CODE_ALIASES:
        return TEST_CODE_ALIASES[code]
    candidate = normalized(value)
    for test_type, aliases in TEST_ALIASES.items():
        if any(normalized(alias) in candidate for alias in aliases):
            return test_type
    return value.strip() or "unknown"


def decimal_degrees(value: str | None) -> float | None:
    """Convert decimal or Japanese degree-minute-second text to degrees."""
    if value is None:
        return None
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
    if len(numbers) >= 3:
        degrees, minutes, seconds = map(float, numbers[:3])
        sign = -1 if degrees < 0 else 1
        return sign * (abs(degrees) + minutes / 60 + seconds / 3600)
    number = safe_number(value)
    return float(number) if isinstance(number, (int, float)) else None


def coordinate_from_parts(values: dict[str, str], field: str) -> float | None:
    """Build a coordinate from decimal text or degree/minute/second fields."""
    direct = decimal_degrees(find_value(values, field))
    if direct is not None:
        return direct
    base = {"latitude": "緯度", "longitude": "経度"}.get(field, field)
    parts = [values.get(normalized(f"{base}_{part}")) for part in ("度", "分", "秒")]
    if all(part is not None for part in parts):
        return decimal_degrees(" ".join(str(part) for part in parts))
    return None


def context_values(root: ET.Element) -> dict[str, str]:
    """Collect borehole-level values while excluding repeated test records."""
    values: dict[str, str] = {}
    for child in root:
        if normalized(local_name(child.tag)) == normalized("試験情報"):
            continue
        for key, value in descendant_values(child).items():
            values.setdefault(key, value)
    return values


def has_strength_content(element: ET.Element) -> bool:
    """Return whether an element looks like a strength-test record."""
    content = normalized(text_of(element))
    return any(normalized(hint) in content for hint in STRENGTH_HINTS) or any(
        find_value(descendant_values(element), field) is not None
        for field in ("c_total", "phi_total", "c_effective", "phi_effective", "qu")
    )


def is_strength_record(element: ET.Element) -> bool:
    """Return whether one test-information element contains useful values."""
    values = descendant_values(element)
    fields = ("c_total", "shear_strength_total", "phi_total", "c_effective", "shear_strength_effective", "phi_effective", "qu", "test_type")
    return any(find_value(values, field) is not None for field in fields)


def candidate_records(root: ET.Element) -> list[ET.Element]:
    """Select likely sample/test containers without assuming one XML hierarchy."""
    test_records = [
        element for element in root.iter()
        if normalized(local_name(element.tag)) == normalized("試験情報")
        and is_strength_record(element)
    ]
    if test_records:
        return test_records
    records: list[ET.Element] = []
    for element in root.iter():
        if element is root:
            continue
        name = normalized(local_name(element.tag))
        if any(token in name for token in ("sample", "試料", "strength", "強度", "triaxial", "三軸", "shear", "せん断")):
            if has_strength_content(element):
                records.append(element)
    if records:
        # Prefer the most specific test container when a sample wraps it.
        return [record for record in records if not any(
            descendant is not record and descendant in records
            for descendant in record.iter()
        )]
    return [root] if has_strength_content(root) else []


def make_row(root_values: dict[str, str], record: ET.Element, xml_file: str) -> dict[str, object]:
    """Build one output row from borehole-level and record-level values."""
    values = dict(root_values)
    record_fields = {
        "sample_id", "sample_no", "depth_top", "depth_bottom", "depth_mid", "sample_z",
        "soil_name", "soil_code", "test_type", "test_condition_code", "test_condition",
        "wet_density", "dry_density", "particle_density", "water_content", "void_ratio",
        "degree_of_saturation", "gravel_fraction", "sand_fraction", "silt_fraction",
        "clay_fraction", "maximum_particle_size", "d10", "d50", "uniformity_coefficient",
        "liquid_limit", "plastic_limit", "plasticity_index",
        "c_total", "shear_strength_total", "phi_total",
        "c_effective", "shear_strength_effective", "phi_effective", "qu",
    }
    for field in record_fields:
        values.pop(normalized(field), None)
    values.update(descendant_values(record))
    row: dict[str, object] = {column: None for column in OUTPUT_COLUMNS}
    row["xml_file"] = xml_file
    for field in OUTPUT_COLUMNS:
        if field in FIELD_ALIASES:
            raw = find_value(values, field)
            row[field] = safe_number(raw) if field not in {
                "boring_id", "sample_id", "soil_name", "soil_code", "test_type",
                "test_condition_code", "test_condition",
            } else raw
    condition = values.get(normalized("せん断試験条件")) or values.get(normalized("せん断試験条件コード"))
    if condition:
        row["test_type"] = test_type_from_text(condition)
    elif row["qu"] is not None:
        row["test_type"] = "一軸圧縮試験"
    else:
        row["test_type"] = test_type_from_text(text_of(record))
    if row["depth_mid"] is None and row["depth_top"] is not None and row["depth_bottom"] is not None:
        try:
            row["depth_mid"] = (float(row["depth_top"]) + float(row["depth_bottom"])) / 2
        except (TypeError, ValueError):
            row["depth_mid"] = None
    if row["boring_id"] is None:
        row["boring_id"] = xml_file.rsplit(".", 1)[0]
    row["latitude"] = coordinate_from_parts(values, "latitude")
    row["longitude"] = coordinate_from_parts(values, "longitude")
    if row["sample_z"] is None and row["surface_z"] is not None and row["depth_mid"] is not None:
        row["sample_z"] = float(row["surface_z"]) - float(row["depth_mid"])
    return row


def parse_xml(path: Path) -> FileResult:
    """Parse one XML file and return zero or more strength rows."""
    try:
        # KuniJiban files commonly declare Shift_JIS; decode explicitly before
        # handing text to ElementTree, which otherwise rejects multibyte XML.
        content = path.read_bytes()
        try:
            xml_text = content.decode("shift_jis")
        except UnicodeDecodeError:
            xml_text = content.decode("utf-8-sig")
        root = ET.fromstring(xml_text)
        root_values = context_values(root)
        rows = [make_row(root_values, record, path.name) for record in candidate_records(root)]
        for element in root.iter():
            UNKNOWN_TAGS[local_name(element.tag)] += 1
        return FileResult(rows=rows)
    except (ET.ParseError, OSError, ValueError, TypeError) as exc:
        return FileResult(rows=[], error=f"{type(exc).__name__}: {exc}")


def configure_logging(log_dir: Path) -> tuple[logging.Logger, logging.Logger]:
    """Create separate error and unknown-tag loggers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    error_logger = logging.getLogger("strength.error")
    unknown_logger = logging.getLogger("strength.unknown")
    for logger, filename in ((error_logger, "error.log"), (unknown_logger, "unknown_tags.log")):
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return error_logger, unknown_logger


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write summary metrics as a two-column CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def build_summary(total_files: int, processed_files: int, error_files: int, rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate file, test-condition, and parameter counts."""
    test_counts = Counter(str(row.get("test_type") or "unknown") for row in rows)
    summary: dict[str, object] = {
        "xml_files_read": total_files,
        "xml_files_processed": processed_files,
        "xml_files_error": error_files,
        "borings_with_strength_data": len({row.get("boring_id") for row in rows}),
        "strength_records": len(rows),
    }
    for test_type in ("UU", "CU", "CUb", "CD", "一軸圧縮試験"):
        summary[f"test_type_{test_type}"] = test_counts[test_type]
    summary["test_type_other"] = sum(count for name, count in test_counts.items() if name not in {"UU", "CU", "CUb", "CD", "一軸圧縮試験"})
    counted_fields = (
                "c_total", "shear_strength_total", "phi_total", "c_effective",
                "shear_strength_effective", "phi_effective", "qu",
        "wet_density", "dry_density", "particle_density", "water_content",
        "void_ratio", "degree_of_saturation", "gravel_fraction", "sand_fraction",
        "silt_fraction", "clay_fraction", "maximum_particle_size", "d10", "d50",
        "uniformity_coefficient", "liquid_limit", "plastic_limit", "plasticity_index",
    )
    for field in counted_fields:
        summary[f"present_{field}"] = sum(row.get(field) is not None for row in rows)
    return summary


def run(input_dir: Path, output_dir: Path, log_dir: Path) -> dict[str, object]:
    """Process all XML files and write CSV, summary, and logs."""
    error_logger, unknown_logger = configure_logging(log_dir)
    paths = sorted(input_dir.rglob("*.xml")) if input_dir.is_dir() else []
    rows: list[dict[str, object]] = []
    processed = 0
    errors = 0
    for path in paths:
        result = parse_xml(path)
        if result.error:
            errors += 1
            error_logger.error("%s: %s", path.name, result.error)
            continue
        processed += 1
        rows.extend(result.rows)
    for tag, count in sorted(UNKNOWN_TAGS.items()):
        unknown_logger.info("%s\t%d", tag, count)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "strength_data.csv", index=False, encoding="utf-8-sig")
    summary = build_summary(len(paths), processed, errors, rows)
    write_summary(summary, output_dir / "summary.csv")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run extraction and print a compact summary."""
    args = parse_args(argv)
    summary = run(args.input_dir, args.output_dir, args.log_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


