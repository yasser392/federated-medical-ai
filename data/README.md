# Data

No data files are stored in this repository.

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset that ships with scikit-learn and is
loaded in memory via `sklearn.datasets.load_breast_cancer()` (no download, no network access):

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 real-valued features computed from digitized fine-needle-aspirate images (mean, standard error and "worst" value of 10 nucleus characteristics) |
| Classes | malignant (212), benign (357) |
| Encoding in this project | **malignant = 1** (positive class), benign = 0 (sklearn uses the opposite) |

Source: W. N. Street, W. H. Wolberg, O. L. Mangasarian, *Nuclear feature extraction for breast tumor
diagnosis*, 1993; UCI Machine Learning Repository.

Split protocol (see `src/data/`):

1. A stratified 20 % **held-out test set** (114 samples) is reserved first.
2. The remaining 455 samples are partitioned across hospitals (IID or Dirichlet non-IID).
3. Each hospital keeps 20 % of its share as a **local validation set**.
4. Standardization statistics are computed from training data only (federated sufficient statistics for FL).
