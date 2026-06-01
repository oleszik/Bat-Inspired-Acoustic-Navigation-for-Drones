"""
Debug unsafe forward cases for action policy v3.

Unsafe scene definition:
- action is MOVE_FORWARD_FAST or MOVE_FORWARD_SLOW
- and any front-group sector (front_left/front/front_right) has true obstacle.

This script is robust to common alternative column names and missing diagnostics.
Missing optional diagnostic columns are created with NaN and reported as warnings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ROOT = Path("datasets/synthetic_echoes_multisector_v1")
SCENE_CSV = DATASET_ROOT / "action_policy_v3_scene_summary.csv"
PRED_CSV = DATASET_ROOT / "multisector_echo_validity_predictions.csv"
LABELS_CSV = DATASET_ROOT / "labels.csv"

OUT_CSV = DATASET_ROOT / "action_policy_v3_unsafe_forward_cases.csv"
OUT_JSON = DATASET_ROOT / "action_policy_v3_unsafe_forward_cases.json"

SECTORS = ["left", "front_left", "front", "front_right", "right"]
FRONT_GROUP = ["front_left", "front", "front_right"]
FORWARD_ACTIONS = {"MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW"}


def warn(msg: str) -> None:
    print(f"WARNING: {msg}")


def resolve_col(df: pd.DataFrame, alternatives: list[str], required: bool = True) -> str | None:
    for c in alternatives:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {alternatives}")
    return None


def canonical_sector(x: object) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "frontleft": "front_left",
        "frontright": "front_right",
        "front_l": "front_left",
        "front_r": "front_right",
    }
    return mapping.get(s, s)


def safe_float(x: object) -> float | None:
    return None if pd.isna(x) else float(x)


def ensure_optional_columns(df: pd.DataFrame, cols: list[str], missing_list: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
            missing_list.append(c)
    return df


def main() -> None:
    for p in [SCENE_CSV, PRED_CSV, LABELS_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    scene_df = pd.read_csv(SCENE_CSV)
    pred_df = pd.read_csv(PRED_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    # Resolve essential columns with alternatives.
    scene_sample_col = resolve_col(scene_df, ["sample_id"], required=True)
    scene_action_col = resolve_col(scene_df, ["recommended_action_v3", "recommended_action"], required=True)

    pred_sample_col = resolve_col(pred_df, ["sample_id"], required=True)
    pred_sector_col = resolve_col(pred_df, ["sector"], required=True)
    pred_state_col = resolve_col(pred_df, ["predicted_state", "state"], required=True)
    pred_dist_col = resolve_col(pred_df, ["predicted_distance_m", "distance_pred_m"], required=False)
    pred_true_obs_col = resolve_col(pred_df, ["true_has_obstacle", "has_obstacle"], required=False)
    pred_true_dist_col = resolve_col(pred_df, ["true_distance_m", "distance_m"], required=False)

    label_sample_col = resolve_col(labels_df, ["sample_id"], required=True)
    label_sector_col = resolve_col(labels_df, ["sector"], required=True)
    label_true_obs_col = resolve_col(labels_df, ["has_obstacle", "true_has_obstacle"], required=True)
    label_true_dist_col = resolve_col(labels_df, ["distance_m", "true_distance_m"], required=False)

    # Normalize ids and sector names.
    scene_df = scene_df.copy()
    pred_df = pred_df.copy()
    labels_df = labels_df.copy()
    scene_df["sample_id_norm"] = scene_df[scene_sample_col].astype(str)
    pred_df["sample_id_norm"] = pred_df[pred_sample_col].astype(str)
    labels_df["sample_id_norm"] = labels_df[label_sample_col].astype(str)

    pred_df["sector_norm"] = pred_df[pred_sector_col].map(canonical_sector)
    labels_df["sector_norm"] = labels_df[label_sector_col].map(canonical_sector)

    pred_sector_values = set(pred_df["sector_norm"].dropna().unique().tolist())
    label_sector_values = set(labels_df["sector_norm"].dropna().unique().tolist())
    for sec in SECTORS:
        if sec not in pred_sector_values:
            warn(f"Prediction CSV missing sector rows for '{sec}'.")
        if sec not in label_sector_values:
            warn(f"Labels CSV missing sector rows for '{sec}'.")

    # Standardize canonical columns in predictions.
    pred_df["predicted_state_norm"] = pred_df[pred_state_col].astype(str).str.strip().str.upper()
    if pred_dist_col is None:
        pred_df["predicted_distance_m_norm"] = np.nan
    else:
        pred_df["predicted_distance_m_norm"] = pd.to_numeric(pred_df[pred_dist_col], errors="coerce")

    if pred_true_obs_col is None:
        pred_df["true_has_obstacle_pred"] = np.nan
    else:
        pred_df["true_has_obstacle_pred"] = pd.to_numeric(pred_df[pred_true_obs_col], errors="coerce")

    if pred_true_dist_col is None:
        pred_df["true_distance_m_pred"] = np.nan
    else:
        pred_df["true_distance_m_pred"] = pd.to_numeric(pred_df[pred_true_dist_col], errors="coerce")

    # Standardize canonical columns in labels (ground truth source).
    labels_df["true_has_obstacle_label"] = pd.to_numeric(labels_df[label_true_obs_col], errors="coerce").fillna(0).astype(int)
    if label_true_dist_col is None:
        labels_df["true_distance_m_label"] = np.nan
        warn("labels.csv missing true distance column; true_distance_m will be NaN where not available.")
    else:
        labels_df["true_distance_m_label"] = pd.to_numeric(labels_df[label_true_dist_col], errors="coerce")

    # Required diagnostic columns we want in output; missing ones should become NaN.
    missing_optional_cols: list[str] = []
    optional_pred_cols = [
        "echo_validity_probability",
        "matched_filter_peak_exists",
        "matched_filter_distance_m",
        "selected_mode",
        "confidence",
        "peak_snr",
        "peak_prominence",
        "peak_width",
        "noise_floor",
        "strongest_peak_value",
        "first_noise_floor_peak_value",
    ]
    optional_label_cols = [
        "reflection_strength",
        "noise_level",
        "surface_absorption",
        "has_secondary_reflection",
    ]
    pred_df = ensure_optional_columns(pred_df, optional_pred_cols, missing_optional_cols)
    labels_df = ensure_optional_columns(labels_df, optional_label_cols, missing_optional_cols)
    if missing_optional_cols:
        warn(f"Missing optional diagnostic columns filled with NaN: {sorted(set(missing_optional_cols))}")

    # Compute front-group obstacle truth per scene from labels (authoritative).
    gt_front = (
        labels_df[labels_df["sector_norm"].isin(FRONT_GROUP)]
        .groupby("sample_id_norm", as_index=False)["true_has_obstacle_label"]
        .max()
        .rename(columns={"true_has_obstacle_label": "gt_true_any_front_group_obstacle"})
    )

    scene = scene_df.merge(gt_front, on="sample_id_norm", how="left")
    if "true_any_front_group_obstacle" in scene.columns:
        scene["true_any_front_group_obstacle_final"] = pd.to_numeric(
            scene["true_any_front_group_obstacle"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        scene["true_any_front_group_obstacle_final"] = pd.to_numeric(
            scene["gt_true_any_front_group_obstacle"], errors="coerce"
        ).fillna(0).astype(int)

    # Find unsafe scenes.
    scene_actions = scene[scene_action_col].astype(str)
    unsafe_mask = scene_actions.isin(FORWARD_ACTIONS) & (scene["true_any_front_group_obstacle_final"] == 1)
    unsafe_scenes = scene.loc[unsafe_mask, ["sample_id_norm", scene_action_col]].copy()
    unsafe_scenes = unsafe_scenes.rename(columns={scene_action_col: "recommended_action"})
    unsafe_sample_ids = unsafe_scenes["sample_id_norm"].tolist()

    # Join predictions with labels for rich diagnostics.
    labels_subset = labels_df[
        [
            "sample_id_norm",
            "sector_norm",
            "true_has_obstacle_label",
            "true_distance_m_label",
            "reflection_strength",
            "noise_level",
            "surface_absorption",
            "has_secondary_reflection",
        ]
    ]
    merged = pred_df.merge(labels_subset, on=["sample_id_norm", "sector_norm"], how="left")
    merged["true_has_obstacle_final"] = pd.to_numeric(merged["true_has_obstacle_label"], errors="coerce")
    merged["true_distance_m_final"] = pd.to_numeric(merged["true_distance_m_label"], errors="coerce")

    unsafe_rows = merged[merged["sample_id_norm"].isin(unsafe_sample_ids)].copy()
    unsafe_rows["sector_norm"] = pd.Categorical(unsafe_rows["sector_norm"], categories=SECTORS, ordered=True)
    unsafe_rows = unsafe_rows.sort_values(["sample_id_norm", "sector_norm"]).reset_index(drop=True)

    # Attach action for each row.
    unsafe_rows = unsafe_rows.merge(unsafe_scenes, on="sample_id_norm", how="left")

    # Identify failed front-group sectors (false negatives).
    unsafe_rows["front_group_false_negative"] = (
        unsafe_rows["sector_norm"].isin(FRONT_GROUP)
        & (unsafe_rows["true_has_obstacle_final"] == 1)
        & (unsafe_rows["predicted_state_norm"] != "OBSTACLE")
    ).astype(int)

    # Build detailed sector-level output table with exact requested columns.
    out_df = pd.DataFrame(
        {
            "sample_id": unsafe_rows["sample_id_norm"],
            "recommended_action": unsafe_rows["recommended_action"],
            "sector": unsafe_rows["sector_norm"].astype(str),
            "true_has_obstacle": unsafe_rows["true_has_obstacle_final"],
            "true_distance_m": unsafe_rows["true_distance_m_final"],
            "predicted_state": unsafe_rows["predicted_state_norm"],
            "predicted_distance_m": unsafe_rows["predicted_distance_m_norm"],
            "echo_validity_probability": pd.to_numeric(unsafe_rows["echo_validity_probability"], errors="coerce"),
            "matched_filter_peak_exists": pd.to_numeric(unsafe_rows["matched_filter_peak_exists"], errors="coerce"),
            "matched_filter_distance_m": pd.to_numeric(unsafe_rows["matched_filter_distance_m"], errors="coerce"),
            "selected_mode": unsafe_rows["selected_mode"],
            "confidence": unsafe_rows["confidence"],
            "peak_snr": pd.to_numeric(unsafe_rows["peak_snr"], errors="coerce"),
            "peak_prominence": pd.to_numeric(unsafe_rows["peak_prominence"], errors="coerce"),
            "peak_width": pd.to_numeric(unsafe_rows["peak_width"], errors="coerce"),
            "noise_floor": pd.to_numeric(unsafe_rows["noise_floor"], errors="coerce"),
            "strongest_peak_value": pd.to_numeric(unsafe_rows["strongest_peak_value"], errors="coerce"),
            "first_noise_floor_peak_value": pd.to_numeric(unsafe_rows["first_noise_floor_peak_value"], errors="coerce"),
            "reflection_strength": pd.to_numeric(unsafe_rows["reflection_strength"], errors="coerce"),
            "noise_level": pd.to_numeric(unsafe_rows["noise_level"], errors="coerce"),
            "surface_absorption": pd.to_numeric(unsafe_rows["surface_absorption"], errors="coerce"),
            "has_secondary_reflection": pd.to_numeric(unsafe_rows["has_secondary_reflection"], errors="coerce"),
        }
    )
    out_df.to_csv(OUT_CSV, index=False)

    # Per-scene diagnosis and prints.
    failed_sector_counts: dict[str, int] = {
        "front_left_false_negative": 0,
        "front_false_negative": 0,
        "front_right_false_negative": 0,
    }
    unsafe_action_counts = (
        unsafe_scenes["recommended_action"].value_counts().to_dict()
        if not unsafe_scenes.empty
        else {}
    )

    per_scene_diagnosis: list[dict[str, object]] = []

    print(f"\nTotal unsafe forward scenes: {len(unsafe_sample_ids)}")
    print(f"Sample IDs: {unsafe_sample_ids}")

    for sid in unsafe_sample_ids:
        scene_rows = out_df[out_df["sample_id"] == sid].copy()
        if scene_rows.empty:
            continue
        action = str(scene_rows["recommended_action"].iloc[0])

        front_rows = scene_rows[scene_rows["sector"].isin(FRONT_GROUP)].copy()
        failed_rows = front_rows[
            (pd.to_numeric(front_rows["true_has_obstacle"], errors="coerce") == 1)
            & (front_rows["predicted_state"].astype(str).str.upper() != "OBSTACLE")
        ].copy()
        failed_sectors = failed_rows["sector"].astype(str).tolist()

        for fs in failed_sectors:
            failed_sector_counts[f"{fs}_false_negative"] = failed_sector_counts.get(f"{fs}_false_negative", 0) + 1

        min_true_distance_front_group = safe_float(
            pd.to_numeric(front_rows["true_distance_m"], errors="coerce").min()
        )
        max_noise_level_front_group = safe_float(
            pd.to_numeric(front_rows["noise_level"], errors="coerce").max()
        )
        min_reflection_strength_front_group = safe_float(
            pd.to_numeric(front_rows["reflection_strength"], errors="coerce").min()
        )
        any_secondary_reflection_front_group = bool(
            (pd.to_numeric(front_rows["has_secondary_reflection"], errors="coerce").fillna(0) > 0).any()
        )
        max_echo_validity_probability_front_group = safe_float(
            pd.to_numeric(front_rows["echo_validity_probability"], errors="coerce").max()
        )
        max_peak_snr_front_group = safe_float(
            pd.to_numeric(front_rows["peak_snr"], errors="coerce").max()
        )
        max_peak_prominence_front_group = safe_float(
            pd.to_numeric(front_rows["peak_prominence"], errors="coerce").max()
        )

        weak = (min_reflection_strength_front_group is not None) and (min_reflection_strength_front_group < 0.05)
        far = (min_true_distance_front_group is not None) and (min_true_distance_front_group > 1.8)
        noisy = (max_noise_level_front_group is not None) and (max_noise_level_front_group > 0.03)
        secondary = any_secondary_reflection_front_group

        if max_echo_validity_probability_front_group is None:
            prob_diag = "missing"
        elif max_echo_validity_probability_front_group < 0.1:
            prob_diag = "very_low"
        elif max_echo_validity_probability_front_group < 0.9:
            prob_diag = "near_threshold"
        else:
            prob_diag = "high"

        if failed_rows.empty:
            matched_filter_failed = False
        else:
            matched_filter_failed = bool(
                (pd.to_numeric(failed_rows["matched_filter_peak_exists"], errors="coerce").fillna(0) == 0).all()
            )

        if matched_filter_failed and weak:
            likely_feature = "Forward lockout when front-group matched_filter_peak_exists=0 and low reflection_strength"
        elif noisy:
            likely_feature = "Noise-aware uncertainty guard before forward motion"
        elif prob_diag == "near_threshold":
            likely_feature = "Obstacle-probability margin band before forward motion"
        else:
            likely_feature = "SNR/prominence-based front uncertainty lockout"

        print("\nUnsafe scene:")
        print(f"  sample_id: {sid}")
        print(f"  recommended_action: {action}")
        print(f"  failed sectors: {failed_sectors}")
        print(f"  weak_reflection: {weak}, far: {far}, high_noise: {noisy}, secondary_reflection: {secondary}")
        print(
            f"  echo_validity_probability: {prob_diag}, matched_filter_failed_completely: {matched_filter_failed}"
        )
        print(f"  likely_feature_to_catch: {likely_feature}")

        per_scene_diagnosis.append(
            {
                "sample_id": sid,
                "recommended_action": action,
                "failed_front_group_sectors": failed_sectors,
                "min_true_distance_front_group": min_true_distance_front_group,
                "max_noise_level_front_group": max_noise_level_front_group,
                "min_reflection_strength_front_group": min_reflection_strength_front_group,
                "any_secondary_reflection_front_group": any_secondary_reflection_front_group,
                "max_echo_validity_probability_front_group": max_echo_validity_probability_front_group,
                "max_peak_snr_front_group": max_peak_snr_front_group,
                "max_peak_prominence_front_group": max_peak_prominence_front_group,
                "weak_reflection": weak,
                "far_obstacle": far,
                "high_noise": noisy,
                "probability_diagnosis": prob_diag,
                "matched_filter_failed_completely": matched_filter_failed,
                "likely_feature_to_catch": likely_feature,
            }
        )

    short_interpretation = (
        "Unsafe forward cases are front-group false negatives driven by weak echoes where matched-filter peaks are missing "
        "and echo-validity confidence is low; add stricter forward lockout for front UNCERTAIN + low-evidence conditions."
        if unsafe_sample_ids
        else "No unsafe forward scenes found."
    )

    report = {
        "total_unsafe_forward_scenes": int(len(unsafe_sample_ids)),
        "unsafe_sample_ids": unsafe_sample_ids,
        "unsafe_action_counts": {k: int(v) for k, v in unsafe_action_counts.items()},
        "failed_sector_counts": {k: int(v) for k, v in failed_sector_counts.items()},
        "per_scene_diagnosis": per_scene_diagnosis,
        "missing_optional_columns_filled_with_nan": sorted(set(missing_optional_cols)),
        "short_interpretation": short_interpretation,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")


if __name__ == "__main__":
    main()
