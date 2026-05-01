# 🚗 Vehicle Insurance Fraud Detection - Multi-Modal Pipeline

A comprehensive multi-modal fraud detection system combining **Tabular (CatBoost)**, **Temporal (LSTM)**, and **Graph (GNN)** models for vehicle insurance fraud detection.

## 📋 Project Overview

**Dataset**: ~15,000 insurance claims with 45+ features
- **Tabular data**: Policy, claim, driver demographics
- **Telematics data**: Driving behavior (speed, braking, acceleration, etc.)
- **Target**: Binary classification (fraud/no fraud) - 6% fraud rate

---

## 🚀 Quick Setup

### Prerequisites
- Python 3.9+
- Virtual environment (conda or venv)
- Git

### 1. Clone & Setup Environment

```bash
cd ~/Workspace/projects/vehicle_fraud_detection
python3 -m venv fraud_env
source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Preprocessing (Required for all models)

```bash
cd notebooks
jupyter notebook 01_preprocessing.ipynb
# Run all cells to generate: ../data/processed.csv
```

**Output**: `data/processed.csv` (15,420 rows × 52 columns)
- All features are numeric (scaled/encoded)
- Target variable: `fraudfound_p` (0 or 1)
- Zero missing values

---

## 👥 Model-Specific Setup

---

## 🎯 **Model 1: CatBoost (Tabular)**

### Use Case
Baseline gradient boosting model using engineered features directly.

### Data Requirements
✅ **Ready to use**: `data/processed.csv`
- 38 numeric features (scaled)
- 12 categorical features (label-encoded)
- Direct feature-to-fraud mapping

### Setup
```bash
# Install additional dependencies
pip install catboost scikit-learn pandas numpy

# Create notebook
cd notebooks
jupyter notebook 02_tabular_model.ipynb
```

### Code Template
```python
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# Load processed data
df = pd.read_csv('../data/processed.csv')
X = df.drop('fraudfound_p', axis=1)
y = df['fraudfound_p']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train CatBoost
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    verbose=0,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

### Key Files
- Input: `data/processed.csv`
- Output: `models/catboost_model.pkl`

---

## 🧠 **Model 2: LSTM (Temporal)**

### Use Case
Sequence modeling of driving behavior over time to detect temporal fraud patterns.

### Data Requirements
⚠️ **NOT ready yet** - Requires preprocessing:

**Expected format**: `(num_drivers, sequence_length, num_features)`
- Example: (10,000 drivers, 30 timesteps, 10 telematics features)

### Setup
```bash
# Install additional dependencies
pip install tensorflow keras torch numpy pandas scikit-learn

# Create notebook
cd notebooks
jupyter notebook 03_temporal_model.ipynb
```

### Data Preparation (LSTM-specific)
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load RAW data (not processed.csv)
df_raw = pd.read_csv('../data/fraud_oracle_with_telematics.csv')

# Extract telematics columns
telematics_cols = [
    'avg_speed_kmph', 'max_speed_kmph', 'hard_brakes_per_trip',
    'rapid_acceleration_events', 'trip_duration_minutes', 'distance_km',
    'night_driving_ratio', 'urban_driving_ratio', 'harsh_cornering_events',
    'idle_time_minutes'
]

# Option 1: If data has multiple trips per driver
# Group by driver_id, sort by timestamp, create 30-step windows
# Shape: (num_drivers, 30, 10_features)

# Option 2: Use aggregated features as synthetic sequences
# Expand processed.csv features into sequences

# Load the saved scaler from preprocessing
import joblib
scaler = joblib.load('../models/scaler.pkl')  # You'll need to save this in preprocessing
X_lstm = scaler.transform(sequences)

# Reshape for LSTM
X_lstm = X_lstm.reshape((num_samples, sequence_length, num_features))
```

### Code Template
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### 📝 TODO for LSTM developer
1. Extract timestamp-based sequences from raw data
2. Create sliding windows (30 trips per driver)
3. Reshape to (num_drivers, 30, 10)
4. Apply scaling
5. Save `scaler.pkl` for reproducibility
6. Train LSTM model

### Key Files
- Input: `data/fraud_oracle_with_telematics.csv` + processed features
- Output: `models/lstm_model.h5`

---

## 📊 **Model 3: GNN (Graph Neural Network)**

### Use Case
Entity relationship modeling - detect fraud rings and connected fraudsters.

### Data Requirements
⚠️ **Partially ready** - Needs graph structure:

**Graph components**:
- **Nodes**: Drivers, Vehicles, Locations, Claims
- **Edges**: Driver-Vehicle, Driver-Location, Driver-Claim relationships
- **Node features**: Use processed.csv features

### Setup
```bash
# Install additional dependencies
pip install torch torch-geometric dgl networkx pandas numpy scikit-learn

# Create notebook
cd notebooks
jupyter notebook 04_graph_model.ipynb
```

### Data Preparation (GNN-specific)
```python
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

# Load processed features
df = pd.read_csv('../data/processed.csv')

# Load raw data for relationships
df_raw = pd.read_csv('../data/fraud_oracle_with_telematics.csv')

# Step 1: Create node mapping
driver_ids = df_raw['policynumber'].unique()
vehicle_ids = df_raw['make'].unique()
location_ids = df_raw['accidentarea'].unique()

driver_map = {id: idx for idx, id in enumerate(driver_ids)}
vehicle_map = {id: idx for idx, id in enumerate(vehicle_ids)}
location_map = {id: idx for idx, id in enumerate(location_ids)}

# Step 2: Create node features tensor
x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float)

# Step 3: Create edges (relationships)
edges = []
# Driver -> Vehicle edges
for idx, row in df_raw.iterrows():
    driver_node = driver_map[row['policynumber']]
    vehicle_node = vehicle_map[row['make']]
    edges.append([driver_node, vehicle_node])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Step 4: Create target labels
y = torch.tensor(df['fraudfound_p'].values, dtype=torch.long)

# Step 5: Create PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
```

### Code Template
```python
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class FraudDetectorGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Train
model = FraudDetectorGNN(num_features, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

### 📝 TODO for GNN developer
1. Map entities (driver, vehicle, location, claim) to nodes
2. Define relationships as edges (driver-vehicle, driver-location, etc.)
3. Use processed.csv features as node attributes
4. Create PyTorch Geometric Data object
5. Implement GNN architecture (GCN, GraphSAGE, etc.)
6. Train and evaluate

### Key Files
- Input: `data/processed.csv` + `data/fraud_oracle_with_telematics.csv`
- Output: `models/gnn_model.pth`

---

## 📂 Directory Structure

```
vehicle_fraud_detection/
├── data/
│   ├── fraud_oracle_with_telematics.csv    (Raw data, 15K rows)
│   ├── processed.csv                       (Preprocessed, all numeric)
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── catboost_model.pkl
│   ├── lstm_model.h5
│   ├── gnn_model.pth
│   └── scaler.pkl                          (For LSTM/GNN)
├── notebooks/
│   ├── 01_preprocessing.ipynb              (✅ COMPLETE)
│   ├── 02_tabular_model.ipynb              (CatBoost - Ready to start)
│   ├── 03_temporal_model.ipynb             (LSTM - Data prep needed)
│   ├── 04_graph_model.ipynb                (GNN - Graph structure needed)
│   └── 05_fusion.ipynb                     (Ensemble - After all models)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Data Specifications

### Processed Data (processed.csv)
| Aspect | Details |
|--------|---------|
| Shape | 15,420 rows × 52 columns |
| Features | 38 numeric (scaled) + 12 categorical (encoded) + 2 IDs |
| Target | `fraudfound_p` (0=no fraud, 1=fraud) |
| Fraud Rate | 5.99% (923 fraud cases) |
| Missing Values | 0 |
| Format | All numeric (int64, float64) |

### Feature Categories
- **Numeric (Scaled)**: age, deductible, driverrating, days_policy_claim, etc.
- **Categorical (Encoded)**: month, dayofweek, sex, maritalstatus, policytype, etc.
- **Engineered Features**: fast_claim, high_claim_history, speeding_risk, harsh_driving_index, etc.
- **Telematics**: avg_speed_kmph, max_speed_kmph, harsh_braking_risk, etc.

---

## 🔄 Workflow

```
01_preprocessing.ipynb (COMPLETE ✅)
    ↓ (generates processed.csv)
    ├──→ 02_tabular_model.ipynb (CatBoost)
    ├──→ 03_temporal_model.ipynb (LSTM) + Raw data
    ├──→ 04_graph_model.ipynb (GNN) + Raw data
    ↓
05_fusion.ipynb (Ensemble combining all models)
    ↓
Final predictions with confidence scores
```

---

## 🛠️ Development Guidelines

### For CatBoost Developer:
- ✅ Data is ready to use
- Implement feature importance analysis
- Try different hyperparameters
- Compare with baseline models
- Save best model in `models/catboost_model.pkl`

### For LSTM Developer:
- ⚠️ Need to preprocess sequences from raw data
- Create sliding window sequences (recommended: 30 timesteps)
- Save scaler to `models/scaler.pkl`
- Implement temporal attention mechanisms (optional)
- Save trained model to `models/lstm_model.h5`

### For GNN Developer:
- ⚠️ Need to build graph structure from entity relationships
- Define node types: driver, vehicle, location, claim
- Use processed features for node embeddings
- Implement message passing layers
- Save model to `models/gnn_model.pth`

---

## 📦 Requirements

Create `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
catboost>=1.0.0
tensorflow>=2.10.0
torch>=1.13.0
torch-geometric>=2.2.0
dgl>=0.7.0
networkx>=2.6.0
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📝 Common Issues & Solutions

### Issue: "processed.csv not found"
**Solution**: Run `01_preprocessing.ipynb` first to generate it

### Issue: "LSTM data shape mismatch"
**Solution**: Ensure sequences are shaped as `(num_samples, sequence_length, num_features)`

### Issue: "Graph creation error"
**Solution**: Verify node mapping is unique and edges match node indices

### Issue: Memory error with GNN
**Solution**: Use batch sampling or reduce graph size for initial testing

---

## 🤝 Collaboration

- **CatBoost**: Start with `02_tabular_model.ipynb` (data ready now)
- **LSTM**: Start with `03_temporal_model.ipynb` (prepare sequences first)
- **GNN**: Start with `04_graph_model.ipynb` (build entity relationships first)
- **Fusion**: After all three models complete, integrate in `05_fusion.ipynb`

---

## ✅ Checklist

- [ ] Preprocessing complete (`01_preprocessing.ipynb`)
- [ ] CatBoost model trained (`02_tabular_model.ipynb`)
- [ ] LSTM sequences prepared (`03_temporal_model.ipynb`)
- [ ] GNN graph structure built (`04_graph_model.ipynb`)
- [ ] All models saved in `models/` directory
- [ ] Fusion model created (`05_fusion.ipynb`)
- [ ] Final predictions and evaluation complete

---

## 📞 Support

For questions about:
- **Preprocessing**: Check `01_preprocessing.ipynb`
- **Feature engineering**: See notebook comments and docstrings
- **Data format**: Refer to Data Specifications section above
- **Model architectures**: See model-specific code templates
