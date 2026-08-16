# symmrnet-breast-ultrasound
Wavelet-Enhanced CNN (SymMRNet-Sym2) for breast ultrasound classification - PhD thesis deployment, SUT

# SymMRNet-Sym2 - Breast Ultrasound Classification

Deployment repository for the model reported in the PhD thesis
"Wavelet-Enhanced CNN for Breast Ultrasound Classification under Speckle Noise"
Department of Interdisciplinary Science and Internationalization, Institute of Science, Suranaree
University of Technology, Nakhon Ratchasima, Thailand,

## Deployed model (single source of truth)
- Weights: `symmrnet_symlet2_3blocks.h5`
- Parameters: 565,658
- Input: 64 x 64 x 1 (grayscale)
- Classes: 0 = benign, 1 = malignant (alphabetical order, as trained)
- Test accuracy: 93.90%
- Preprocessing: grayscale -> /255 -> sym2 DWT (symmetric) -> sub-band rescaling
  (cA x1.05, cH x1.02, cV x1.02, cD x1.01) -> IDWT -> crop/pad -> clip. No CLAHE.

The application loads this file only and fails at startup on any parameter-count
mismatch. Earlier VMC-Net weights are archived under `legacy/` and are NOT deployed.

## Intended use
Research and academic demonstration only. Not a medical device.
Not validated for clinical decision-making.
