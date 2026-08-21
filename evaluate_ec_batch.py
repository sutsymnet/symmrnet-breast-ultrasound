"""
evaluate_ec_batch.py - Chapter VII clinical evaluation
======================================================
Runs the DEPLOYED SymMRNet-Sym2 model over the SUT Hospital EC image set
using the exact inference path served by the web prototype.

DATASET (updated after the STEP 0 v2 cleaning pipeline)
------------------------------------------------------
316 images, NOT the 300 originally planned:
    DatasetA 100 | DatasetB 100 | DatasetC 116 (30/20/22/24/20)
The full set is the headline denominator rather than discarding 16 real
patient images to make a table look tidy. A balanced 20-per-cell subset is
reported as a sensitivity analysis instead.

Design decisions baked in (each is defensible at the viva):

* PRIMARY analysis = BI-RADS 2-5, non-Doppler only, decision rule = argmax
  (0.50). Locked before looking at results, identical to Chapter V. Nothing
  is tuned on the EC data.

* BI-RADS 1 (normal, no discrete lesion) is EXCLUDED from the primary
  analysis and reported separately. The model was trained only on lesion
  images; folding "normal" into "benign" would inflate accuracy on cases the
  model never saw during training.

* Colour-Doppler images are EXCLUDED under the SAME rule - outside the
  trained domain. Colour overlays replace the underlying echo data, and
  decode_and_resize() converts to greyscale, so the lesion interior becomes
  mid-grey texture that is not real tissue. The induced error is
  unpredictable in direction, not merely inflationary. The same exclusion is
  enforced at inference time by input_gate.py in the deployed prototype, so
  what is validated matches what is deployed.
  Flagged in _manifest.csv (column `is_doppler`), n = 11 of 316.

* SECONDARY analyses (exploratory, NOT headline numbers):
    - Doppler-included comparison (shows the effect of that exclusion)
    - balanced 20-per-cell subset (shows the effect of the unbalanced C set)
    - threshold sweep over P(malignant)
    - BI-RADS 3 boundary sensitivity (3 -> benign vs 3 -> malignant)
    - per-BI-RADS-level breakdown
    - per-dataset breakdown (checks the 3 collection batches agree)

USAGE
-----
    $ python evaluate_ec_batch.py --data-dir 02_cleaned

_manifest.csv from step0_v2_partC.py is picked up automatically from the data
directory. Without it the Doppler exclusion cannot be applied, and the script
says so loudly rather than silently reporting the wrong primary number.

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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

BIRADS_TO_BINARY = {1: None, 2: "benign", 3: "benign", 4: "malignant", 5: "malignant"}

BALANCED_PER_CELL = 20
BALANCED_SEED = 20260821


# --------------------------------------------------------------- discovery
def scan_directory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        # skip the pipeline's own artefacts (_contact_*.png, _manifest.csv ...)
        if path.name.startswith("_"):
            continue
        match = re.search(r"bi[-_ ]?rads[-_ ]?([1-5])", str(path), re.IGNORECASE)
        if match is None:
            print(f"  [skip] cannot infer BI-RADS from path: {path}")
            continue
        rel = path.relative_to(root)
        rows.append({
            "filepath": str(path),
            "rel_path": str(rel),
            "birads": int(match.group(1)),
            "dataset": rel.parts[0] if len(rel.parts) > 1 else "all",
        })
    return pd.DataFrame(rows)


def attach_manifest(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """
    Merge the `is_doppler` flag produced by step0_v2_partC.py.

    Absence is treated as a hard error rather than a silent default: running
    the primary analysis without the Doppler exclusion would report a number
    that does not match the stated protocol, and nothing downstream would
    reveal the discrepancy.
    """
    man = root / "_manifest.csv"
    if not man.exists():
        print(f"\n  !! {man} not found.")
        print("  !! Cannot apply the Doppler exclusion, so the PRIMARY analysis")
        print("  !! would not match the protocol. Run step0_v2_partC.py first,")
        print("  !! or pass --allow-missing-manifest to proceed without it.")
        return None

    m = pd.read_csv(man)
    if "is_doppler" not in m.columns:
        print(f"  !! {man} has no `is_doppler` column")
        return None

    m = m[["rel_path", "is_doppler"]].copy()
    m["rel_path"] = m["rel_path"].str.replace("\\", "/", regex=False)
    df = df.copy()
    df["rel_path"] = df["rel_path"].str.replace("\\", "/", regex=False)
    out = df.merge(m, on="rel_path", how="left")

    n_missing = int(out["is_doppler"].isna().sum())
    if n_missing:
        print(f"  [warn] {n_missing} image(s) not listed in _manifest.csv "
              f"-> treated as non-Doppler")
    out["is_doppler"] = out["is_doppler"].fillna(False).astype(bool)
    print(f"  manifest merged: {int(out['is_doppler'].sum())} Doppler image(s) flagged")
    return out


def balanced_subset(df: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    """Up to `per_cell` images per (dataset, birads), sampled with a fixed seed."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby(["dataset", "birads"], sort=True):
        if len(g) <= per_cell:
            keep.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=per_cell, replace=False)
            keep.append(g.loc[np.sort(idx)])
    return pd.concat(keep).sort_index()


# -------------------------------------------------------------- inference
def run_inference(model, df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    p_benign, p_malignant, failures = [], [], []
    paths = df["filepath"].tolist()
    for start in range(0, len(paths), batch_size):
        chunk = paths[start: start + batch_size]
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
        print(f"\n  {len(failures)} image(s) failed to decode:")
        for p, e in failures[:10]:
            print(f"    {p}: {e}")
    return out


# ---------------------------------------------------------------- metrics
def binary_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    def safe(num, den):
        return float(num / den) if den else float("nan")

    return {
        "n": int(total),
        "confusion_matrix": {
            "rows_true": CLASS_NAMES, "cols_pred": CLASS_NAMES,
            "matrix": cm.tolist(),
        },
        "TP_malignant": int(tp), "TN_benign": int(tn),
        "FP": int(fp), "FN": int(fn),
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
        rows.append({
            "threshold": float(t), "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity_malignant"],
            "specificity": m["specificity_benign"],
            "balanced_accuracy": m["balanced_accuracy"],
        })
    return pd.DataFrame(rows)


def print_metrics(m, indent="  "):
    for k in ("n", "accuracy", "sensitivity_malignant", "specificity_benign",
              "PPV", "NPV", "balanced_accuracy"):
        v = m[k]
        print(f"{indent}{k:<24} {v}" if k == "n" else f"{indent}{k:<24} {v:.4f}")
    print(f"{indent}confusion matrix (rows=true {CLASS_NAMES}, cols=pred):")
    for name, row in zip(CLASS_NAMES, m["confusion_matrix"]["matrix"]):
        print(f"{indent}  {name:<10} {row}")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path(MODEL_FILENAME))
    ap.add_argument("--out-dir", type=Path, default=Path("ec_results"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--allow-missing-manifest", action="store_true",
                    help="run without _manifest.csv (Doppler exclusion disabled)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bar = "=" * 68
    print(bar)
    print("Chapter VII - SUT Hospital EC clinical evaluation")
    print(bar)

    df = scan_directory(args.data_dir)
    if df.empty:
        raise SystemExit("No images found. Check paths / BI-RADS folder names.")
    print(f"\nFound {len(df)} images")
    print(df.groupby(["dataset", "birads"]).size().unstack(fill_value=0), "\n")

    merged = attach_manifest(df, args.data_dir)
    if merged is None:
        if not args.allow_missing_manifest:
            raise SystemExit("Stopping. See message above.")
        df["is_doppler"] = False
        print("  [!] proceeding WITHOUT the Doppler exclusion")
    else:
        df = merged

    print(f"\nLoading {args.model} ...")
    model = load_symmrnet(str(args.model), verify=True)
    print(f"  verified: input {model.input_shape}, {model.count_params():,} params\n")

    print("Running inference ...")
    df = run_inference(model, df, args.batch_size)
    df["label_binary"] = df["birads"].map(BIRADS_TO_BINARY)
    df.to_csv(args.out_dir / "per_image_predictions.csv", index=False)

    results = {
        "model_file": str(args.model),
        "n_images_total": int(len(df)),
        "n_doppler_excluded": int(df["is_doppler"].sum()),
        "birads_to_binary_mapping": {str(k): v for k, v in BIRADS_TO_BINARY.items()},
        "decision_rule_primary": "argmax over softmax (== P(malignant) >= 0.50)",
        "primary_inclusion": "BI-RADS 2-5, non-Doppler",
    }

    scored = df["label_binary"].notna() & df["pred_argmax"].notna()

    # ---------------- PRIMARY
    prim = df[scored & (~df["is_doppler"])]
    print("\n" + bar)
    print(f"PRIMARY - BI-RADS 2-5, non-Doppler, argmax (n = {len(prim)})")
    print(bar)
    m = binary_metrics(prim["label_binary"], prim["pred_argmax"])
    results["primary"] = m
    print_metrics(m)

    y_bin = (prim["label_binary"] == "malignant").astype(int).to_numpy()
    if y_bin.min() != y_bin.max():
        auc = float(roc_auc_score(y_bin, prim["p_malignant"]))
        results["primary"]["AUC"] = auc
        print(f"  {'AUC':<24} {auc:.4f}")
        fpr, tpr, thr = roc_curve(y_bin, prim["p_malignant"])
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
            args.out_dir / "roc_primary.csv", index=False)

    # ---------------- SENSITIVITY A: Doppler included
    incl = df[scored]
    n_dop = len(incl) - len(prim)
    print("\n" + "-" * 68)
    print(f"SENSITIVITY A - Doppler INCLUDED (n = {len(incl)}, +{n_dop} images)")
    print("-" * 68)
    if n_dop:
        m_incl = binary_metrics(incl["label_binary"], incl["pred_argmax"])
        results["sensitivity_doppler_included"] = m_incl
        print(f"  primary (excluded)  acc {m['accuracy']:.4f}  "
              f"sens {m['sensitivity_malignant']:.4f}  "
              f"spec {m['specificity_benign']:.4f}")
        print(f"  Doppler included    acc {m_incl['accuracy']:.4f}  "
              f"sens {m_incl['sensitivity_malignant']:.4f}  "
              f"spec {m_incl['specificity_benign']:.4f}")
        print("  ^ a large gap means the exclusion is doing real work and must")
        print("    be stated prominently, not buried in the methods section")
    else:
        print("  no Doppler images in BI-RADS 2-5")

    # ---------------- SENSITIVITY B: balanced subset
    bal = balanced_subset(prim, BALANCED_PER_CELL, BALANCED_SEED)
    print("\n" + "-" * 68)
    print(f"SENSITIVITY B - balanced <= {BALANCED_PER_CELL}/cell "
          f"(n = {len(bal)}, seed {BALANCED_SEED})")
    print("-" * 68)
    print(pd.crosstab(bal["dataset"], bal["birads"], margins=True).to_string())
    m_bal = binary_metrics(bal["label_binary"], bal["pred_argmax"])
    results["sensitivity_balanced_subset"] = {
        **m_bal, "per_cell": BALANCED_PER_CELL, "seed": BALANCED_SEED,
    }
    print(f"\n  full (unbalanced)   acc {m['accuracy']:.4f}  "
          f"sens {m['sensitivity_malignant']:.4f}  "
          f"spec {m['specificity_benign']:.4f}")
    print(f"  balanced subset     acc {m_bal['accuracy']:.4f}  "
          f"sens {m_bal['sensitivity_malignant']:.4f}  "
          f"spec {m_bal['specificity_benign']:.4f}")

    # ---------------- SECONDARY 1: BI-RADS 1
    b1 = df[(df["birads"] == 1) & df["pred_argmax"].notna() & (~df["is_doppler"])]
    if len(b1):
        share_benign = float((b1["pred_argmax"] == "benign").mean())
        results["secondary_birads1_out_of_scope"] = {
            "n": int(len(b1)),
            "predicted_benign_rate": share_benign,
            "predicted_malignant_rate": 1 - share_benign,
            "mean_p_malignant": float(b1["p_malignant"].mean()),
            "note": "Normal studies, no discrete lesion. Outside the trained "
                    "domain. Reported descriptively; excluded from accuracy.",
        }
        print("\n" + "-" * 68)
        print(f"SECONDARY - BI-RADS 1, out of trained domain (n = {len(b1)})")
        print("-" * 68)
        print(f"  predicted benign     {share_benign:.4f}")
        print(f"  predicted malignant  {1 - share_benign:.4f}")
        print(f"  mean P(malignant)    {b1['p_malignant'].mean():.4f}")

    # ---------------- SECONDARY 2: threshold sweep
    print("\n" + "-" * 68)
    print("SECONDARY - threshold sweep (exploratory, NOT the headline result)")
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
    print("SECONDARY - BI-RADS 3 boundary sensitivity")
    print("-" * 68)
    alt = df[scored & (~df["is_doppler"])].copy()
    alt["label_binary"] = alt["birads"].map({**BIRADS_TO_BINARY, 3: "malignant"})
    m_alt = binary_metrics(alt["label_binary"], alt["pred_argmax"])
    results["secondary_birads3_as_malignant"] = m_alt
    print(f"  BI-RADS 3 -> benign     acc {m['accuracy']:.4f}  "
          f"sens {m['sensitivity_malignant']:.4f}  "
          f"spec {m['specificity_benign']:.4f}")
    print(f"  BI-RADS 3 -> malignant  acc {m_alt['accuracy']:.4f}  "
          f"sens {m_alt['sensitivity_malignant']:.4f}  "
          f"spec {m_alt['specificity_benign']:.4f}")

    # ---------------- SECONDARY 4: per level / per dataset
    per_level = (
        df[df["pred_argmax"].notna() & (~df["is_doppler"])]
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
    print("SECONDARY - per BI-RADS level (non-Doppler)")
    print("-" * 68)
    print(per_level.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    per_ds = []
    for ds, g in prim.groupby("dataset"):
        row = {"dataset": ds}
        row.update({k: v for k, v in
                    binary_metrics(g["label_binary"], g["pred_argmax"]).items()
                    if k in ("n", "accuracy", "sensitivity_malignant",
                             "specificity_benign")})
        per_ds.append(row)
    per_ds = pd.DataFrame(per_ds)
    per_ds.to_csv(args.out_dir / "per_dataset.csv", index=False)
    results["secondary_per_dataset"] = per_ds.to_dict(orient="records")
    print("\n" + "-" * 68)
    print("SECONDARY - per collection batch (checks the 3 sets agree)")
    print("-" * 68)
    print(per_ds.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    small = per_ds[per_ds["n"] < 60]
    if len(small):
        print("\n  [note] batches with n < 60 have wide confidence intervals;")
        print("         report them with that caveat rather than bare numbers.")

    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("\n" + bar)
    print(f"Written to {args.out_dir}/")
    print("  per_image_predictions.csv  per_birads_level.csv  per_dataset.csv")
    print("  threshold_sweep.csv  roc_primary.csv  summary.json")
    print(bar)


if __name__ == "__main__":
    main()
