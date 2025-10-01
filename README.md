# Intrusion Detection System (IDS) — GCN on KNN Graphs 🔐

A graph-based Intrusion Detection System that detects network attacks (Normal / DoS / DDoS) by modeling network flows as graphs and training a Graph Convolutional Network (GCN).  
This implementation uses **KNN graphs (k = 10)** to connect similar flows and a GCN to learn graph-structured patterns — a setup that fits well because attack flows are often correlated with surrounding traffic.

---

## 🚀 Project Summary (step-by-step)

### 1. Raw packets → Sessions
- Raw network captures are in `.pcap` format (single packets).  
- A single packet rarely contains enough context to detect attacks. So packets are **merged into sessions**.
- **Session** = unique tuple of `(source IP, destination IP, protocol)` plus aggregated flow features (per-session statistics).
- Each session row can contain many features (example: ~82 features per session).

### 2. Flow extraction (CICFlowMeter)
- I used **CICFlowMeter** (tool from the Canadian Institute for Cybersecurity) to convert `.pcap` files into flow/session CSVs — the same approach used to build the CIC-IDS2017 dataset.

### 3. Dataset
- Training / evaluation used **CIC-IDS2017** (Canadian Institute for Cybersecurity) — contains labeled benign and various attack flows (DoS, DDoS, etc.).

### 4. Graph construction (KNN, k = 10)
- Convert flow rows into a graph using **K-Nearest Neighbors (KNN)** on flow feature vectors.
- Each flow is a node; edges connect to the `k = 10` nearest flows (by chosen distance metric).
- Intuition: attack-related flows tend to be similar/clustered; graph connectivity helps GCN exploit these relations.

### 5. GCN training
- Train a **Graph Convolutional Network (GCN)** on the KNN graph to classify nodes (flows) as Normal / DoS / DDoS.
- The model learns both feature and neighborhood (structural) information — useful for detecting correlated attack behavior.
- Current reported performance: **~97% accuracy for DoS/DDoS detection** (as measured on the evaluation split used in this project).

---



## 📂 Repository structure
```bash
Dharam13/ (root)
├── Plots/ # training/eval plots (loss, accuracy, confusion matrix, ...)
├── Saved_Model/ # trained model checkpoints (GCN weights)
├── Testing Files/ # sample .pcap / flow CSVs used for testing (real DoS and benign traffic)
├── .gitattributes
├── Model.py # model training script (GCN training pipeline)
├── Model_Training_Output.txt # logged training output & metrics
├── Prediction.py # prediction script: load saved model and run inference on flows/graphs
├── Prediction_2.py # alternative prediction/benchmark script
└── features_alignment.py # feature cleaning / alignment utilities (ensure train/test features match)

```

# 🛡️ DoS/DDoS Detection using Graph Convolutional Networks (GCN)

A machine learning pipeline for detecting Denial-of-Service (DoS) and Distributed Denial-of-Service (DDoS) attacks using Graph Convolutional Networks (GCN) on network flow data. This project leverages KNN graphs to model relationships between network flows and achieves ~97% detection accuracy.

## 📋 Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Feature Alignment](#2-feature-alignment)
  - [3. Training the Model](#3-training-the-model)
  - [4. Running Predictions](#4-running-predictions)
- [Project Structure](#project-structure)
- [Results](#results)
- [Configuration](#configuration)
- [Credits](#credits)
- [Future Work](#future-work)


## ✨ Features

- **GCN-based Classification**: Utilizes Graph Convolutional Networks for network flow classification
- **KNN Graph Construction**: Builds k-nearest neighbor graphs (k=10) to capture flow relationships
- **High Accuracy**: Achieves ~97% accuracy on DoS/DDoS detection
- **CIC-IDS2017 Support**: Compatible with the widely-used CIC-IDS2017 dataset
- **Modular Pipeline**: Separate scripts for training, testing, and prediction
- **Comprehensive Logging**: Training metrics and results saved automatically

## 🔧 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- CICFlowMeter (for PCAP to flow conversion)

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
torch-geometric>=2.0.0
networkx>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dos-ddos-detection-gcn.git
   cd dos-ddos-detection-gcn
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   # OR
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install CICFlowMeter** (for PCAP conversion)
   ```bash
   # Follow instructions at: https://github.com/ahlashkari/CICFlowMeter
   ```

## 🚀 Usage

### 1. Data Preparation

**Option A: Using existing CIC-IDS2017 CSVs**
- Download CIC-IDS2017 dataset from [CIC website](https://www.unb.ca/cic/datasets/ids-2017.html)
- Place CSV files in the `Data/` directory

**Option B: Converting PCAP files**
```bash
# Convert PCAP to flows using CICFlowMeter
cicflowmeter -f input.pcap -c output.csv
```

### 2. Feature Alignment

Ensure feature columns match training expectations:

```bash
python features_alignment.py
```

This script:
- Cleans and normalizes feature names
- Handles missing values
- Ensures consistent feature ordering
- Saves aligned data to `Data/aligned/`

### 3. Training the Model

```bash
python Model.py
```

**Training process:**
- Loads preprocessed flow data
- Constructs KNN graph with k=10
- Trains GCN model with specified hyperparameters
- Saves model checkpoints to `Saved_Model/`
- Logs training metrics to `Model_Training_Output.txt`
- Generates plots in `Plots/`

**Customize training parameters** by editing `Model.py`:
```python
# Example parameters
k = 10              # KNN neighbors
epochs = 100        # Training epochs
learning_rate = 0.01
hidden_dim = 64
```

### 4. Running Predictions

**Single prediction script:**
```bash
python Prediction.py
```

**Batch testing script:**
```bash
python Prediction_2.py
```

**Using test files:**
```bash
# Test on sample DoS flows
python Prediction.py --input Testing_Files/dos_sample.csv

# Test on benign traffic
python Prediction.py --input Testing_Files/benign_sample.csv
```

**Output:**
- Predicted labels (DoS, DDoS, Benign)
- Confidence scores
- Performance metrics
- Results saved to `Prediction_Output.txt`
```

## 📈 Results

The GCN model demonstrates strong performance on DoS/DDoS detection:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | ~97% |
| **DoS Detection Rate** | 98.2% |
| **DDoS Detection Rate** | 96.5% |
| **False Positive Rate** | 2.1% |

**Performance Details:**
- Training logs: `Model_Training_Output.txt`
- Confusion matrices: `Plots/confusion_matrix.png`
- Training curves: `Plots/training_curve.png`

## ⚙️ Configuration

### KNN Graph Parameters

Modify `k` value in `Model.py`:
```python
# Smaller k: focuses on tighter local neighborhoods
k = 5

# Larger k: captures broader context
k = 20
```

**Recommendations:**
- k=5-10: Best for distinct attack patterns
- k=15-20: Better for capturing gradual transitions
- Default k=10 provides balanced performance



## Credits

**Dataset:**
- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) - Canadian Institute for Cybersecurity
- Generated using [CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter)

**Libraries & Frameworks:**
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - GNN library
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities
- [NetworkX](https://networkx.org/) - Graph analysis

## 🔮 Future Work

- [ ] **Extended Attack Coverage**: Include more attack families (Port Scan, Web Attacks, Brute Force)
- [ ] **Dynamic Graphs**: Implement time-windowed graphs for evolving attack detection
- [ ] **Real-time Detection**: Deploy model as REST API for live traffic analysis
- [ ] **Ensemble Methods**: Combine multiple GNN architectures



