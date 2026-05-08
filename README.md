# Vehicle Fraud Detection

An end-to-end machine learning notebook project for detecting potentially fraudulent vehicle insurance claims. The project combines traditional claim attributes with synthetic telematics signals, then trains multiple complementary models and fuses their learned embeddings into a final fraud-risk classifier.

The pipeline is intentionally split into notebooks so each modeling branch can be inspected, rerun, and evaluated independently:

1. Preprocess raw claim and telematics data.
2. Train a tabular CatBoost fraud model.
3. Train a temporal LSTM over ordered claim/telematics sequences.
4. Train a graph neural network over claim relationship edges.
5. Merge all branch embeddings and train a final fusion DNN.

## Project Goals

Vehicle insurance fraud is a rare-event classification problem: most claims are legitimate, while fraudulent claims are costly and difficult to identify. This project approaches that imbalance by combining several views of the same claim:

- **Tabular view:** structured insurance fields such as vehicle category, policy type, claimant demographics, accident metadata, and claim history.
- **Telematics view:** driving behavior signals such as speed, braking, acceleration, night driving, distance, and idle time.
- **Temporal view:** recent neighboring claims ordered by available time fields, modeled as short sequences.
- **Graph view:** relationships between claims that share meaningful insurance attributes, modeled with a graph convolutional network.
- **Fusion view:** a final neural network that learns from all branch embeddings and model probabilities.

The result is a research-style fraud detection pipeline that emphasizes interpretability of stages, feature engineering, embedding generation, and threshold tuning for high-recall fraud screening.

## Repository Structure

```text
.
|-- data/
|   |-- fraud_oracle_with_telematics.csv   # Raw source dataset with claim and telematics fields
|   |-- preprocessed.csv                   # Cleaned, encoded, scaled canonical dataset
|   |-- tabular_embeddings.csv             # CatBoost branch embeddings and probabilities
|   |-- temporal_embeddings.csv            # LSTM branch embeddings and probabilities
|   `-- graph_embeddings.csv               # GCN branch embeddings and probabilities
|-- notebooks/
|   |-- 01_preprocessing.ipynb             # Raw data cleaning and feature engineering
|   |-- 02_tabular_model.ipynb             # CatBoost or sklearn fallback model
|   |-- 03_temporal_model.ipynb            # LSTM sequence model
|   |-- 04_graph_model.ipynb               # PyTorch Geometric GCN model
|   `-- 05_fusion.ipynb                    # Final embedding fusion DNN and threshold tuning
|-- requirements.txt                       # Python dependencies
|-- LICENSE                                # MIT license
`-- README.md
```

The local environment directory `fraud_env/` is ignored by `.gitignore` and should not be treated as source code.

## Dataset

The raw file is `data/fraud_oracle_with_telematics.csv`.

Current raw dataset shape:

- Rows: `15,420`
- Columns: `43`
- Target column: `FraudFound_P`, normalized during preprocessing to `fraudfound_p`

Important raw feature groups include:

- Claim timing: `Month`, `WeekOfMonth`, `DayOfWeek`, `MonthClaimed`, `WeekOfMonthClaimed`
- Claim and policy details: `Fault`, `PolicyType`, `VehicleCategory`, `VehiclePrice`, `BasePolicy`
- Driver and policyholder fields: `Sex`, `MaritalStatus`, `Age`, `AgeOfPolicyHolder`
- Claim history: `PastNumberOfClaims`, `NumberOfSuppliments`, `Days_Policy_Accident`, `Days_Policy_Claim`
- Investigation indicators: `PoliceReportFiled`, `WitnessPresent`, `AgentType`
- Telematics fields: `avg_speed_kmph`, `max_speed_kmph`, `hard_brakes_per_trip`, `rapid_acceleration_events`, `trip_duration_minutes`, `distance_km`, `night_driving_ratio`, `urban_driving_ratio`, `harsh_cornering_events`, `idle_time_minutes`

After preprocessing, the target distribution in the checked notebook output is:

```text
legitimate claims: 14,497
fraud claims:        923
```

This means fraud is only about 6 percent of the dataset, so accuracy alone is not a useful success metric. The notebooks report ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrices.

## Pipeline Overview

### 1. Preprocessing

Notebook: `notebooks/01_preprocessing.ipynb`

Input:

- `data/fraud_oracle_with_telematics.csv`

Output:

- `data/preprocessed.csv`

Main steps:

- Standardizes column names and text values.
- Creates a stable `claim_id` from `PolicyNumber`.
- Drops administrative identifiers such as `PolicyNumber` and `RepNumber` from model features.
- Fills numeric missing values with medians.
- Fills categorical missing values with modes.
- Converts range-like insurance fields into numeric values.
- Engineers telematics and claim-risk features:
  - `speeding_risk`
  - `speed_volatility`
  - `harsh_braking_risk`
  - `harsh_acceleration_risk`
  - `harsh_cornering_risk`
  - `harsh_driving_index`
  - `high_night_driving`
  - `high_urban_driving`
  - `fast_claim`
  - `high_claim_history`
  - `claim_driving_risk`
- Encodes categorical columns with `LabelEncoder`.
- Scales numeric columns with `StandardScaler`, while preserving `claim_id`, `fraudfound_p`, and `year`.

Checked output shape:

```text
data/preprocessed.csv: 15,420 rows x 53 columns
```

### 2. Tabular Model

Notebook: `notebooks/02_tabular_model.ipynb`

Input:

- `data/preprocessed.csv`

Output:

- `data/tabular_embeddings.csv`

Model:

- Primary: `CatBoostClassifier`
- Fallback: `HistGradientBoostingClassifier` if CatBoost is unavailable

Main steps:

- Splits data with stratification using `test_size=0.2` and `random_state=42`.
- Trains a balanced classifier for the rare fraud target.
- Uses CatBoost leaf indexes as compact tabular embeddings.
- Normalizes embedding columns for fusion stability.
- Saves 16 embedding dimensions, raw score, tabular fraud probability, and target.

Checked output shape:

```text
data/tabular_embeddings.csv: 15,420 rows x 20 columns
```

Representative checked metrics from the notebook:

```text
ROC-AUC: 0.8330
PR-AUC:  0.2192
fraud recall at 0.5 threshold: 0.79
fraud precision at 0.5 threshold: 0.15
```

### 3. Temporal Model

Notebook: `notebooks/03_temporal_model.ipynb`

Input:

- `data/preprocessed.csv`

Output:

- `data/temporal_embeddings.csv`

Model:

- PyTorch LSTM

Main steps:

- Selects temporal and telematics columns.
- Sorts claims by available time fields: `year`, `month`, `weekofmonth`, `dayofweek`, and `claim_id`.
- Builds fixed-length rolling windows of length `5`.
- Trains an LSTM with weighted cross entropy.
- Saves 16 temporal embedding dimensions and temporal fraud probability.

Checked output shape:

```text
data/temporal_embeddings.csv: 15,420 rows x 19 columns
```

The current temporal model is useful as a branch signal, but its checked standalone metrics are weak compared with the tabular model. This is expected because the source data has one row per claim, so the notebook constructs local sequence context from neighboring claims rather than true per-policy time series.

### 4. Graph Model

Notebook: `notebooks/04_graph_model.ipynb`

Input:

- `data/preprocessed.csv`

Output:

- `data/graph_embeddings.csv`

Model:

- PyTorch Geometric GCN

Main steps:

- Treats each claim as a graph node.
- Builds edges between claims that share important relationship attributes.
- Uses a bounded number of edges per group to avoid giant fully connected components.
- Trains a graph convolutional network with class-weighted loss.
- Saves 32 graph embedding dimensions and graph fraud probability.

Relationship columns used when available:

- `make`
- `accidentarea`
- `fault`
- `policytype`
- `vehiclecategory`
- `basepolicy`
- `ageofvehicle`
- `pastnumberofclaims`
- `high_claim_history`

Checked graph size:

```text
nodes: 15,420
edges: 906,604
```

Checked output shape:

```text
data/graph_embeddings.csv: 15,420 rows x 35 columns
```

### 5. Fusion Model

Notebook: `notebooks/05_fusion.ipynb`

Inputs:

- `data/tabular_embeddings.csv`
- `data/temporal_embeddings.csv`
- `data/graph_embeddings.csv`

Output:

- The notebook trains and evaluates the final model in memory. It does not currently save a model artifact.

Model:

- PyTorch feed-forward DNN

Main steps:

- Merges all branch outputs on `claim_id` and `fraudfound_p`.
- Drops IDs, target columns, and raw-score leakage columns.
- Creates additional fusion features:
  - `avg_prob`
  - `max_prob`
  - `min_prob`
  - `weighted_score`
  - `tab_x_graph`
  - `tab_x_temp`
  - `tab_graph_diff`
  - `tab_temp_diff`
  - `final_hint`
- Scales fusion features.
- Trains a weighted binary classifier with early stopping on ROC-AUC.
- Evaluates default threshold behavior.
- Runs threshold tuning for:
  - precision-first selection under a minimum recall target
  - minimum business cost selection
  - optional focal-loss comparisons

Checked merged shape:

```text
15,420 rows x 70 columns before dropping/feature expansion
```

Representative checked final metrics:

```text
Best ROC-AUC: 0.8291
Test Accuracy @0.5: 0.6848
Precision @0.5: 0.1406
Recall @0.5: 0.8324
F1 @0.5: 0.2406
PR-AUC: 0.2259
```

The threshold analysis shows why recall-focused tuning matters for fraud detection. For example, the checked run selected a precision-first threshold of `0.54` while maintaining recall at `0.80`.

## Installation

### Prerequisites

Install the following before running the notebooks:

- Python 3.10 or newer
- `pip`
- Jupyter Notebook or JupyterLab
- A C/C++ build toolchain only if your platform cannot install prebuilt wheels for PyTorch, CatBoost, or PyTorch Geometric

The repository was inspected in an environment with Python `3.13.5`, but some ML packages may have better wheel availability on Python 3.10, 3.11, or 3.12. If installation fails on Python 3.13, create a Python 3.11 environment and retry.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd vehicle_fraud_detection
```

If you already have the project locally, just move into the project directory:

```bash
cd /home/ahan/Workspace/projects/vehicle_fraud_detection
```

### 2. Create a Virtual Environment

Using Python `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Using Conda:

```bash
conda create -n vehicle-fraud python=3.11 -y
conda activate vehicle-fraud
```

### 3. Upgrade Packaging Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

The dependency file includes:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `catboost`
- `torch`
- `torch-geometric`
- `imblearn`
- `ipykernel`

### 5. Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name vehicle-fraud --display-name "Vehicle Fraud Detection"
```

Then select the `Vehicle Fraud Detection` kernel inside Jupyter.

### 6. Start Jupyter

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Open the notebooks from the `notebooks/` directory and run them in order.

## PyTorch Geometric Installation Notes

`torch-geometric` depends on your Python version, PyTorch version, CPU/GPU setup, and operating system. If the normal requirements install fails, install PyTorch first from the official selector for your platform, then install PyTorch Geometric.

CPU-only example:

```bash
pip install torch
pip install torch-geometric
```

If you use CUDA, install the PyTorch build matching your CUDA version before installing `torch-geometric`.

After installation, verify the graph dependency:

```bash
python -c "import torch; import torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

## How to Run the Full Project

Run the notebooks in this exact order:

```text
1. notebooks/01_preprocessing.ipynb
2. notebooks/02_tabular_model.ipynb
3. notebooks/03_temporal_model.ipynb
4. notebooks/04_graph_model.ipynb
5. notebooks/05_fusion.ipynb
```

The dependency chain is:

```text
fraud_oracle_with_telematics.csv
        |
        v
01_preprocessing.ipynb
        |
        v
preprocessed.csv
        |
        +--> 02_tabular_model.ipynb  --> tabular_embeddings.csv
        +--> 03_temporal_model.ipynb --> temporal_embeddings.csv
        +--> 04_graph_model.ipynb    --> graph_embeddings.csv
                                            |
                                            v
                                   05_fusion.ipynb
```

You can rerun an individual branch notebook after preprocessing, but the fusion notebook requires all three embedding files to exist.

## Generated Files

The current project includes generated CSV files in `data/` so the fusion workflow can be inspected immediately. If you rerun notebooks, these files may be overwritten:

- `data/preprocessed.csv`
- `data/tabular_embeddings.csv`
- `data/temporal_embeddings.csv`
- `data/graph_embeddings.csv`

The `.gitignore` also excludes common training outputs such as:

- `notebooks/catboost_info/`
- `models/`
- `*.pkl`
- `*.joblib`
- `*.pt`
- `*.pth`
- notebook checkpoints

## Reproducibility

The notebooks use fixed random seeds where practical:

- `random_state=42` for train/test splits
- `random_seed=42` for CatBoost

PyTorch models can still vary slightly across runs because of initialization, backend behavior, hardware, and package versions. For stricter reproducibility, add explicit `numpy` and `torch` seeds at the top of the PyTorch notebooks and configure deterministic PyTorch behavior.

## Evaluation Guidance

Because the fraud class is rare, use these metrics instead of relying on accuracy:

- **ROC-AUC:** broad ranking quality across thresholds.
- **PR-AUC:** more informative for rare fraud labels.
- **Recall:** percentage of actual frauds caught.
- **Precision:** percentage of flagged claims that are actually fraud.
- **F1 score:** balance between precision and recall.
- **Confusion matrix:** operational view of false alarms and missed frauds.
- **Business cost:** custom threshold selector in the fusion notebook using false-positive and false-negative costs.

In fraud screening, a higher recall threshold may be preferred even if precision is low, because missing an actual fraud can be more expensive than reviewing a legitimate claim.

## Troubleshooting

### `python: command not found`

Use `python3` instead:

```bash
python3 -m venv .venv
python3 -m pip install -r requirements.txt
```

### CatBoost installation fails

Try upgrading packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install catboost
```

The tabular notebook has a sklearn fallback, but CatBoost is recommended for the intended workflow.

### `torch_geometric` import fails

Install PyTorch and PyTorch Geometric in separate steps and verify compatible versions:

```bash
pip install torch
pip install torch-geometric
python -c "import torch_geometric; print('ok')"
```

### Jupyter does not show the environment

Register the environment as a kernel:

```bash
python -m ipykernel install --user --name vehicle-fraud --display-name "Vehicle Fraud Detection"
```

Restart Jupyter after registering the kernel.

### Fusion notebook cannot find embedding files

Run these notebooks first:

```text
01_preprocessing.ipynb
02_tabular_model.ipynb
03_temporal_model.ipynb
04_graph_model.ipynb
```

Then rerun `05_fusion.ipynb`.

## Possible Improvements

Good next engineering improvements include:

- Save trained models and scalers to a `models/` directory.
- Convert notebook logic into reusable Python modules.
- Add a command-line training pipeline.
- Use a validation split for threshold selection instead of selecting thresholds on the test split.
- Add model cards or experiment logs for each branch.
- Add SHAP or CatBoost feature importance reporting for the tabular model.
- Replace synthetic neighboring-claim sequences with true policyholder or vehicle-level time series if available.
- Add unit tests for preprocessing and data validation.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
