"""
evaluate_ec_batch.py - Chapter VII clinical evaluation
======================================================
Runs the DEPLOYED SymMRNet-Sym2 model over the SUT Hospital EC image set
(3 datasets x 100 images = 300, balanced 20 per class across BI-RADS 1-5)
using the exact inference path served by the web prototype.

Design decisions baked in (each is defensible at the viva):

  * PRIMARY analysis = BI-RADS 2-5 only, decision rule = argmax (0.50).
    Locked before looking at results, identical to Chapter V. Nothing is tuned
    on the EC data.

  * BI-RADS 1 (normal, no discrete lesion) is EXCLUDED from the primary
    analysis and reported separately. The model was trained only on lesion
    images; folding "normal" into "benign" would inflate accuracy on cases the
    model never saw during training.

  * SECONDARY analyses (clearly labelled as exploratory, not headline numbers):
      - threshold sweep over P(malignant)
      - BI-RADS 3 boundary sensitivity (3 -> benign vs 3 -> malignant)
      - per-BI-RADS-level breakdown
      - per-dataset breakdown (checks the 3 collection batches agree)

USAGE
-----
Directory layout (default):
    ec_data/
        dataset1/BIRADS1/*.png   dataset1/BIRADS2/*.png  ...
        dataset2/...
        dataset3/...
    $ python evaluate_ec_batch.py --data-dir ec_data

Or a manifest CSV with columns: filepath,birads[,dataset]
    $ python evaluate_ec_batch.py --manifest ec_manifest.csv

Requires: symmrnet_core.py, symmrnet_symlet2_3blocks.h5,
          tensorflow, PyWavelets, numpy, pandas, scikit-learn
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from symmrnet_core import (
    BENIGN_IDX,
    MALIGNANT_IDX,
    CLASS_NAMES,
    MODEL_FILENAME,
    apply_symlet2_preprocessing_numpy,
    decode_and_resize,
    load_symmrnet,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm.png"}

# Locked mapping. BI-RADS 4 is the biopsy-recommended threshold in ACR
# BI-RADS; 1 is normal and therefore out of the model's trained domain.
BIRADS_TO_BINARY = {1: None, 2: "benign", 3: "benign", 4: "malignant", 5: "malignant"}


# --------------------------------------------------------------- discovery
def scan_directory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = re.search(r"bi[-_ ]?rads[-_ ]?([1-5])", str(path), re.IGNORECASE)
        if match is None:
            print(f"  [skip] cannot infer BI-RADS from path: {path}")
            continue
        rel = path.relative_to(root).parts
        rows.append(
            {
                "filepath": str(path),
                "birads": int(match.group(1)),
                "dataset": rel[0] if len(rel) > 1 else "all",
            }
        )
    return pd.DataFrame(rows)


def load_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"filepath", "birads"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing column(s): {sorted(missing)}")
    if "dataset" not in df.columns:
        df["dataset"] = "all"
    df["birads"] = df["birads"].astype(int)
    return df


# -------------------------------------------------------------- inference
def run_inference(model, df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    p_benign, p_malignant, failures = [], [], []
    paths = df["filepath"].tolist()

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        raws, ok_idx = [], []
        for i, p in enumerate(chunk):
            try:
                raws.append(decode_and_resize(Path(p).read_bytes()))
                ok_idx.append(i)
            except Exception as exc:  # noqa: BLE001
                failures.append((p, str(exc)))

        probs_chunk = np.full((len(chunk), 2), np.nan, dtype=np.float64)
        if raws:
            batch = apply_symlet2_preprocessing_numpy(np.stack(raws, axis=0))
            probs_chunk[ok_idx] = model.predict(batch, verbose=0)

        p_benign.extend(probs_chunk[:, BENIGN_IDX])
        p_malignant.extend(probs_chunk[:, MALIGNANT_IDX])
        print(f"  processed {min(start + batch_size, len(paths))}/{len(paths)}")

    out = df.copy()
    out["p_benign"] = p_benign
    out["p_malignant"] = p_malignant
    out["pred_argmax"] = np.where(
        out["p_malignant"] > out["p_benign"], "malignant", "benign"
    )
    out.loc[out["p_malignant"].isna(), "pred_argmax"] = None

    if failures:
        print(f"\n  ⚠️ {len(failures)} image(s) failed to decode:")
        for p, e in failures[:10]:
            print(f"     {p}: {e}")
    return out


# ---------------------------------------------------------------- metrics
def binary_metrics(y_true, y_pred):
    """y_true / y_pred are arrays of 'benign' | 'malignant'."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    def safe(num, den):
        return float(num / den) if den else float("nan")

    return {
        "n": int(total),
        "confusion_matrix": {
            "rows_true": CLASS_NAMES,
            "cols_pred": CLASS_NAMES,
            "matrix": cm.tolist(),
        },
        "TP_malignant": int(tp),
        "TN_benign": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": safe(tp + tn, total),
        "sensitivity_malignant": safe(tp, tp + fn),
        "specificity_benign": safe(tn, tn + fp),
        "PPV": safe(tp, tp + fp),
        "NPV": safe(tn, tn + fn),
        "balanced_accuracy": (safe(tp, tp + fn) + safe(tn, tn + fp)) / 2,
    }


def threshold_sweep(y_true, p_malignant, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = np.where(p_malignant >= t, "malignant", "benign")
        m = binary_metrics(y_true, pred)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": m["accuracy"],
                "sensitivity": m["sensitivity_malignant"],
                "specificity": m["specificity_benign"],
                "balanced_accuracy": m["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--model", type=Path, default=Path(MODEL_FILENAME))
    ap.add_argument("--out-dir", type=Path, default=Path("ec_results"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if (args.data_dir is None) == (args.manifest is None):
        ap.error("give exactly one of --data-dir or --manifest")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Chapter VII — SUT Hospital EC clinical evaluation")
    print("=" * 68)

    df = scan_directory(args.data_dir) if args.data_dir else load_manifest(args.manifest)
    if df.empty:
        raise SystemExit("No images found. Check paths / BI-RADS folder names.")

    print(f"\nFound {len(df)} images")
    print(df.groupby(["dataset", "birads"]).size().unstack(fill_value=0), "\n")

    print(f"Loading {args.model} …")
    model = load_symmrnet(str(args.model), verify=True)
    print(f"  ✅ verified: input {model.input_shape}, {model.count_params():,} params\n")

    print("Running inference …")
    df = run_inference(model, df, args.batch_size)
    df["label_binary"] = df["birads"].map(BIRADS_TO_BINARY)
    df.to_csv(args.out_dir / "per_image_predictions.csv", index=False)

    results = {
        "model_file": str(args.model),
        "n_images_total": int(len(df)),
        "birads_to_binary_mapping": {str(k): v for k, v in BIRADS_TO_BINARY.items()},
        "decision_rule_primary": "argmax over softmax (equivalent to P(malignant) >= 0.50)",
    }

    # ---------------- PRIMARY: BI-RADS 2-5, argmax
    prim = df[df["label_binary"].notna() & df["pred_argmax"].notna()]
    print("\n" + "=" * 68)
    print(f"PRIMARY ANALYSIS — BI-RADS 2-5, argmax (n = {len(prim)})")
    print("=" * 68)
    m = binary_metrics(prim["label_binary"], prim["pred_argmax"])
    results["primary"] = m
    for k in ("accuracy", "sensitivity_malignant", "specificity_benign", "PPV", "NPV",
              "balanced_accuracy"):
        print(f"  {k:<24} {m[k]:.4f}")
    print(f"  confusion matrix (rows=true {CLASS_NAMES}, cols=pred):")
    for name, row in zip(CLASS_NAMES, m["confusion_matrix"]["matrix"]):
        print(f"    {name:<10} {row}")

    y_bin = (prim["label_binary"] == "malignant").astype(int).to_numpy()
    if y_bin.min() != y_bin.max():
        auc = float(roc_auc_score(y_bin, prim["p_malignant"]))
        results["primary"]["AUC"] = auc
        print(f"  {'AUC':<24} {auc:.4f}")
        fpr, tpr, thr = roc_curve(y_bin, prim["p_malignant"])
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
            args.out_dir / "roc_primary.csv", index=False
        )

    # ---------------- SECONDARY 1: BI-RADS 1 behaviour
    b1 = df[(df["birads"] == 1) & df["pred_argmax"].notna()]
    if len(b1):
        share_benign = float((b1["pred_argmax"] == "benign").mean())
        results["secondary_birads1_out_of_scope"] = {
            "n": int(len(b1)),
            "predicted_benign_rate": share_benign,
            "predicted_malignant_rate": 1 - share_benign,
            "mean_p_malignant": float(b1["p_malignant"].mean()),
            "note": "Normal studies, no discrete lesion. Outside the trained "
                    "domain (training data contained lesion images only). "
                    "Reported descriptively; excluded from accuracy.",
        }
        print("\n" + "-" * 68)
        print(f"SECONDARY — BI-RADS 1, out of trained domain (n = {len(b1)})")
        print("-" * 68)
        print(f"  predicted benign     {share_benign:.4f}")
        print(f"  predicted malignant  {1 - share_benign:.4f}")
        print(f"  mean P(malignant)    {b1['p_malignant'].mean():.4f}")

    # ---------------- SECONDARY 2: threshold sweep
    print("\n" + "-" * 68)
    print("SECONDARY — threshold sweep (exploratory, NOT the headline result)")
    print("-" * 68)
    sweep = threshold_sweep(prim["label_binary"], prim["p_malignant"].to_numpy())
    sweep.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = sweep.loc[sweep["balanced_accuracy"].idxmax()]
    results["secondary_threshold_sweep"] = {
        "best_balanced_accuracy_threshold": float(best["threshold"]),
        "best_balanced_accuracy": float(best["balanced_accuracy"]),
        "note": "Post-hoc on EC data. Must NOT be reported as the operating "
                "point of the deployed system.",
    }

    # ---------------- SECONDARY 3: BI-RADS 3 boundary
    print("\n" + "-" * 68)
    print("SECONDARY — BI-RADS 3 boundary sensitivity")
    print("-" * 68)
    alt = df.copy()
    alt["label_binary"] = alt["birads"].map({**BIRADS_TO_BINARY, 3: "malignant"})
    alt = alt[alt["label_binary"].notna() & alt["pred_argmax"].notna()]
    m_alt = binary_metrics(alt["label_binary"], alt["pred_argmax"])
    results["secondary_birads3_as_malignant"] = m_alt
    print(f"  BI-RADS 3 -> benign     acc {m['accuracy']:.4f}  "
          f"sens {m['sensitivity_malignant']:.4f}  spec {m['specificity_benign']:.4f}")
    print(f"  BI-RADS 3 -> malignant  acc {m_alt['accuracy']:.4f}  "
          f"sens {m_alt['sensitivity_malignant']:.4f}  spec {m_alt['specificity_benign']:.4f}")

    # ---------------- SECONDARY 4: per-level and per-dataset
    per_level = (
        df[df["pred_argmax"].notna()]
        .groupby("birads")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "mean_p_malignant": g["p_malignant"].mean(),
            "pred_malignant_rate": (g["pred_argmax"] == "malignant").mean(),
        }), include_groups=False)
        .reset_index()
    )
    per_level.to_csv(args.out_dir / "per_birads_level.csv", index=False)
    print("\n" + "-" * 68)
    print("SECONDARY — per BI-RADS level")
    print("-" * 68)
    print(per_level.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    per_ds = []
    for ds, g in prim.groupby("dataset"):
        row = {"dataset": ds}
        row.update({k: v for k, v in binary_metrics(g["label_binary"], g["pred_argmax"]).items()
                    if k in ("n", "accuracy", "sensitivity_malignant", "specificity_benign")})
        per_ds.append(row)
    per_ds = pd.DataFrame(per_ds)
    per_ds.to_csv(args.out_dir / "per_dataset.csv", index=False)
    results["secondary_per_dataset"] = per_ds.to_dict(orient="records")
    print("\n" + "-" * 68)
    print("SECONDARY — per collection batch (checks the 3 sets agree)")
    print("-" * 68)
    print(per_ds.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print(f"Written to {args.out_dir}/")
    print("  per_image_predictions.csv  per_birads_level.csv  per_dataset.csv")
    print("  threshold_sweep.csv        roc_primary.csv       summary.json")
    print("=" * 68)


if __name__ == "__main__":
    main()
