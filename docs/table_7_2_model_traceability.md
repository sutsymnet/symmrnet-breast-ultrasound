# Table 7.2 — Deployed model traceability record

Verified fields were read directly from `symmrnet_symlet2_3blocks.h5` and from the
training notebook. `[FILL]` fields are ones only you can supply.

| # | Field | Value | Source of evidence |
|---|---|---|---|
| 1 | Model name | SymMRNet-Symlet2-3Blocks (Wavelet-Sym2, preprocessed) | `model_version` string, notebook cell 11 |
| 2 | Weight file | `symmrnet_symlet2_3blocks.h5` (6.84 MB) | file inspection |
| 3 | Training notebook | `147.4_3Block_TrainValTestDS03.3_SymPreprocess+Sym2Pooling_Epoch12_BS=8.ipynb` | — |
| 4 | Input specification | 64 × 64 × 1, grayscale, float32 ∈ [0, 1] | `model_config` → `batch_shape [null,64,64,1]` |
| 5 | Total parameters | 565,658 (all trainable) | `model.count_params()` |
| 6 | Architecture | 3 × (Conv2D → Symlet2Pooling) → AvgPool 2×2 → Flatten → Dropout 0.5 → Dense 256 (ReLU) → Dropout 0.5 → Dense 2 (softmax) | `model_config` |
| 7 | Conv filter progression | 12 → 32 → 128, all 3×3, `padding='same'`, ReLU | `model_config` |
| 8 | Pooling operation | Fixed Symlet-2 low-pass 2×2 depthwise conv (`padding='SAME'`) → ReLU → 2×2 average pool (`padding='VALID'`); 0 trainable parameters | notebook cell 4 |
| 9 | Optimiser / loss | Adam, lr = 3.0 × 10⁻⁴; categorical cross-entropy | `training_config` |
| 10 | Batch size / epochs | 8 / 12 | notebook cell 2 |
| 11 | Framework | Keras 3.8.0, TensorFlow backend | `.h5` root attributes |
| 12 | Training dataset | *Ultrasound Breast Images for Breast Cancer* (Kaggle), internal split DS03.3 | notebook cell 3 |
| 13 | Split sizes | train 7,212 / val 902 / test 902 | notebook cell 8 output |
| 14 | Class index order | 0 = benign, 1 = malignant (alphabetical, `image_dataset_from_directory`) | notebook cell 27 output |
| 15 | Held-out test accuracy | 0.9390 | notebook cells 16, 27 |
| 16 | Held-out confusion matrix | [[431, 20], [35, 416]] | notebook cell 20 output |
| 17 | Per-class F1 | benign 0.9400 / malignant 0.9380 | notebook cell 27 output |
| 18 | Decision rule (deployed) | `argmax` over softmax, equivalent to P(malignant) ≥ 0.50; fixed a priori, never tuned on evaluation data | notebook cells 19–20 |
| 19 | Deployment preprocessing | grayscale decode → bilinear resize 64×64 → ÷255 → `pywt.dwt2(·,'sym2',mode='symmetric')` → cA×1.05, cH×1.02, cV×1.02, cD×1.01 → `pywt.idwt2` → crop/pad 64×64 → clip [0,1]. **No CLAHE, no augmentation.** | notebook cell 5 |
| 20 | Deployment platform | Streamlit Community Cloud | — |
| 21 | Application entry point | `streamlit_app.py` | repo root |
| 22 | Repository | `github.com/sutsymnet/symmrnet-breast-ultrasound` | — |
| 23 | Public URL | `[FILL — the current app URL]` | — |
| 24 | Commit SHA of evaluated build | `[FILL — git rev-parse --short HEAD]` | — |
| 25 | Deployment date of evaluated build | `[FILL]` | — |
| 26 | Clinical evaluation set | SUT Hospital EC set, 300 images (3 batches × 100; 20 per class per BI-RADS level 1–5) | EC protocol |
| 27 | Reference standard | Radiologist-assigned BI-RADS category | EC protocol |
| 28 | BI-RADS → binary mapping | 2, 3 → benign; 4, 5 → malignant; 1 excluded (normal; outside trained domain) | fixed a priori, §7.x |
| 29 | Primary analysis set | BI-RADS 2–5, n = 240 | — |
| 30 | Ethics approval | `[EC certificate number EC-68-0196 and approval date 6 May 2026]` | — |
| 31 | Superseded deployment | `github.com/dmathsb22/ultrasound-breast-cancer-app` (VMC-Net weights; retired, not used for Chapter VII) | — |
---

## Text to insert under the table

> The prototype loads a single weight file, `symmrnet_symlet2_3blocks.h5`, with
> no fallback path. The loader verifies input shape and parameter count against
> the values in rows 4–5 and raises an error on mismatch, so the application
> cannot silently serve a network other than the one evaluated in Chapter V.
> The preprocessing chain in row 19 is imported from a shared module used
> unchanged by both the web application and the batch evaluation script, so the
> inference path reported in this chapter is by construction identical to the
> one served to users.

---

## Corrections this table forces elsewhere in the thesis

| Location | Currently says | Must become |
|---|---|---|
| §7.6 | Input 224 × 224 | **64 × 64 × 1 grayscale** |
| §7.x / Ch.6 | Prototype deploys the Chapter V model | Now true — but note in the revision log that the previously deployed build loaded `ultrasound_model_vmc_net.h5` and was replaced |
| Eq. 15–16, Eq. 25 | Symlet-2 DWT producing the LL sub-band | Fixed Symlet-2-derived low-pass depthwise filter + ReLU + 2×2 average pooling — an **approximation** of the LL sub-band, not a separable 2-D DWT |
| Ch.3 Table 3.3 | (preprocessing list) | Confirm CLAHE is **absent**; the sub-band rescaling factors 1.05/1.02/1.02/1.01 must be stated explicitly |
| Ch.8 Limitations | — | Add: wavelet pooling is a 2×2 approximation; high-pass sub-bands are discarded at every pooling stage; the sub-band rescaling factors were set heuristically and not ablated |
