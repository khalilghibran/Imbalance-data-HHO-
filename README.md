# HHO and GA for Imbalanced Data Classification

A comprehensive pipeline for handling imbalanced pharmaceutical adverse side effect datasets using Harris Hawks Optimization (HHO) and machine learning techniques.

## Project Overview

This project addresses the challenge of **severe class imbalance** in pharmaceutical side effect datasets (SIDER) by combining:
- **Morgan fingerprints** for molecular feature representation
- **Oversampling techniques** (SMOTE, ADASYN, Random Oversampling)
- **Harris Hawks Optimization (HHO)** for intelligent undersampling
- **Multiple classification models** (RandomForest, XGBoost)

The pipeline evaluates performance using metrics: **F1-Score**, **Balanced Accuracy (BAS)**, and **Geometric Mean (GM)**.

## Dataset

### Source
- **SIDER Database**: Pharmaceutical adverse side effects dataset
- **Path**: `/home/gibannn/kuliah/sem3/paper/SMILES2VEC/data/SIDER/sider.csv`

### Preprocessing Steps
1. **SMILES to Morgan Fingerprints**: Convert molecular structures into 2048-bit binary fingerprints
2. **Binary Label Detection**: Identify and validate binary (0/1) labels
3. **Imbalance Analysis**: Calculate imbalance ratios (IR) for each label
4. **Label Selection**: Choose 9 representative labels (3 highest IR, 3 median IR, 3 lowest IR)

### Selected Labels (by Imbalance Ratio)
Nine target labels are selected for evaluation, covering various organ systems and disease categories from the SIDER database.

## Features

### Morgan Fingerprints
- **Radius**: 2
- **Bits**: 2048
- **Format**: Sparse CSR matrix (uint8 binary format)
- **File**: `X_features_morgan_2048.npz`

### Stored as
- **Combined Dataset**: `combined_dataset.csv` (SMILES + 2048 bit features + 9 labels)
- **Summary**: `imbalance_summary.csv` (IR, positive/negative counts per label)

## Methods

### Stage 1: Baseline Evaluation (No Resampling)
- **Models**: RandomForest, XGBoost
- **CV**: 5-fold Stratified Group K-Fold (prevents data leakage)
- **Results**: `baseline_no_resampling_top2_*.csv`

### Stage 2: Oversampling-Only Refinement
- **Techniques**: SMOTE, ADASYN, Random Oversampling
- **Strategy**: Increase minority class to 1.3× majority class size
- **Models**: RandomForest, XGBoost
- **CV**: 5-fold Stratified Group K-Fold (group-aware)
- **Results**: `oversampling_only_SMOTE_ADASYN_top2_*.csv`
- **Best Per Label**: Selected via GM (F1 and BAS as tie-breakers)

### Stage 3: HHO-Refined Undersampling
Harris Hawks Optimization (HHO) performs intelligent undersampling with adaptive parameter tuning:

#### HHO Algorithm
- **Optimization Target**: Keep ratio of majority samples (0.2–0.9)
- **Population Size**: 12 hawks
- **Iterations**: 25
- **Inner CV**: 3-fold group-aware cross-validation for fitness evaluation
- **Fitness Metric**: GM (Geometric Mean) with F1/BAS tie-breakers

#### Workflow per Fold
1. **Hardness Ranking**: Score samples using classifier margin; keep hardest majority samples
2. **Subset Formation**: Combine selected majority + all minority samples
3. **Oversampling**: Apply winning resampling method (SMOTE/ADASYN/ROS)
4. **Classification**: Train and evaluate on outer fold validation set

#### Results Storage
- **File**: `hho_refinement_results_*.csv`
- **Metrics**: Valid GM, Valid BAS, Valid F1, Best keep ratio, Inner GM fitness

## Results Summary

### Evaluation Metrics
- **F1-Score**: Balance between precision and recall
- **Balanced Accuracy (BAS)**: Unbiased accuracy accounting for class imbalance
- **Geometric Mean (GM)**: √(TPR × TNR) — robust measure for imbalanced classification

### Ranking Strategy
Per-label results ranked by:
1. **Primary**: GM (Geometric Mean)
2. **Secondary**: F1-Score
3. **Tertiary**: Balanced Accuracy

## File Structure

```
├── README.md                                          # This file
├── imbalanced_new.ipynb                               # Main processing pipeline
├── combined_dataset.csv                               # Preprocessed dataset
├── imbalance_summary.csv                              # IR statistics
├── X_features_morgan_2048.npz                         # Morgan fingerprints (sparse)
├── baseline_no_resampling_top2_*.csv                  # Stage 1 results
├── oversampling_only_SMOTE_ADASYN_top2_*.csv         # Stage 2 results
├── hho_refinement_results_*.csv                       # Stage 3 results
└── preprocessing/
    └── Proprocessing.ipynb                            # Data preprocessing notebook
```

## Dependencies

### Core Libraries
```python
pandas, numpy, scipy
rdkit                          # Chemistry: SMILES→Morgan fingerprints
scikit-learn                   # ML: RandomForest, scaling, metrics
imbalanced-learn              # Resampling: SMOTE, ADASYN
xgboost                        # Classification: XGBoost
```

### Installation
```bash
pip install pandas numpy scipy rdkit scikit-learn imbalanced-learn xgboost
```

## Usage

### Run the Full Pipeline
1. Open `imbalanced_new.ipynb` in Jupyter
2. Update `Data_path` to your SIDER dataset location
3. Execute cells sequentially:
   - **Step 1-7**: Data loading and preprocessing
   - **Step 8**: Baseline evaluation
   - **Step 9**: Oversampling refinement
   - **Step 10**: Best combination selection
   - **Step 11**: HHO undersampling optimization

### Custom Parameters
Modify within the notebook:
```python
RANDOM_STATE = 116              # Reproducibility
MORGAN_BITS = 2048             # Fingerprint size
MINORITY_MULTIPLIER = 1.30      # Oversampling target
HHO_ITERATIONS = 25            # Optimization iterations
HHO_POPULATION = 12            # Population size
```

## Key Results & Insights

### Performance Progression
1. **Baseline**: Unbalanced models, lower GM due to class imbalance
2. **Oversampling**: Improved recall on minority class
3. **HHO Refinement**: Balanced undersampling finds optimal keep ratios
   - Typical keep ratios: 0.3–0.7 (retain 30–70% of majority class)
   - Inner CV GM improves model robustness

### Effectiveness Factors
- **Oversampling Method**: Label-dependent (SMOTE/ADASYN often best)
- **Model Selection**: XGBoost typically outperforms RandomForest on rare labels
- **HHO Benefit**: Discovers dataset-specific optimal undersampling without manual tuning

## Reproducibility

- **Random Seed**: `RANDOM_STATE = 116` (set globally)
- **Group-Aware CV**: Uses `StratifiedGroupKFold` to prevent group leakage
- **Balanced Evaluation**: Metrics include zero_division handling for edge cases

## Citation & References

### Methods
- **Morgan Fingerprints**: Rogers & Hahn (2010) — molecular representation
- **SMOTE**: Chawla et al. (2002) — synthetic minority oversampling
- **ADASYN**: He et al. (2008) — adaptive synthetic sampling
- **Harris Hawks Optimization**: Heidari et al. (2019) — nature-inspired metaheuristic

### Dataset
- **SIDER**: Kuhn et al. (2016) — Adverse Drug Reaction database
  - http://sideeffects.embl.de/

