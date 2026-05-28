# STRCDA

Sparse Topological Representation Learning and Dual-View Decoding for circRNA-Disease Association Prediction

---

## Overview

STRCDA is a computational framework for predicting potential circRNA-disease associations by integrating biological similarity information, random walk with restart (RWR), sparse graph representation learning, and XGBoost-based association prediction.

The framework combines local topological refinement and global graph embedding to effectively capture complex interaction patterns between circRNAs and diseases.

---

## Abstract

Circular RNAs (circRNAs) are key regulators in the onset and progression of complex diseases, offering promise as diagnostic and prognostic biomarkers. However, most putative circRNA-disease associations remain experimentally unverified, largely due to the cost and time demands of wet-lab approaches. To bridge this gap, we present STRCDA (Sparse Topological Representation Learning for CircRNA-Disease Associations). The pipeline first constructs fused similarity profiles for circRNAs and diseases by integrating diverse biological attributes. These initial matrices are then refined via random walk with restart to capture local features. Subsequently, a sparse-constrained dual-branch graph autoencoder extracts holistic topological embeddings from the refined local features and the known interaction network. Finally, an XGBoost classifier scores potential circRNA-disease pairs. On the CircR2Disease dataset, STRCDA achieves an AUC of 0.9771 and an AUPR of 0.9826 under five-fold cross-validation. Notably, 18 of the top 20 predicted associations were confirmed by independent experimental evidence, highlighting STRCDA’s efficacy as a robust tool for uncovering circRNA function in disease.

---

## Framework

The STRCDA framework mainly consists of the following components:

1. Multi-source similarity construction and fusion
2. Local topology enhancement via Random Walk with Restart (RWR)
3. Dual-decoder sparse graph autoencoder
4. Sparse regularization with L1 constraint
5. XGBoost-based association prediction

---

## Repository Structure

```text
STRCDA/
│
├── dataset/                    # Benchmark datasets
│
├── feature.py                  # Feature construction and RWR refinement
├── layers.py                   # Graph neural network layers
├── model.py                    # GCN-AE and GCN-VAE architectures
├── optimizer.py                # Model optimization strategies
├── preprocessing.py            # Graph preprocessing utilities
├── evaluation.py               # Performance evaluation and visualization
├── train.py                    # Main training pipeline
│
├── README.md
└── requirements.txt
```

---

## Requirements

The implementation was tested under the following environment:

* Python 3.8
* TensorFlow 1.15
* PyTorch 1.10
* NumPy 1.23
* SciPy 1.9
* scikit-learn 1.1
* XGBoost 1.7

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Datasets

The proposed model is evaluated on the following benchmark datasets:

* CircR2Disease
* CircAtlas
* Circ2Disease
* CircRNADisease

### Input Files

Each dataset directory should contain:

```text
AM.csv                # circRNA-disease association matrix
RNA_sim.csv           # circRNA similarity matrix
Disease_sim.csv       # Disease similarity matrix
```

---

## Usage

### Step 1: Generate refined feature matrices

```bash
python feature.py
```

This step performs:

* similarity fusion,
* random walk with restart (RWR),
* heterogeneous feature construction.

Generated file:

```text
feature.csv
```

---

### Step 2: Train the graph autoencoder

```bash
python train.py
```

This step generates:

* latent node embeddings,
* reconstructed adjacency matrices,
* reconstructed feature matrices.

Generated files:

```text
0-emb_train_rwr
0-data_rec_adj_rwr
0-data_rec_fea_rwr
```

---

### Step 3: Perform association prediction

```bash
python xgboost_classifier.py
```

This step predicts potential circRNA-disease associations using XGBoost.

---

### Step 4: Evaluate model performance

```bash
python evaluation.py
```

This step computes:

* Accuracy (Acc)
* Precision
* Recall
* F1-score
* MCC
* AUC
* AUPR

and generates ROC/PR curves.

---

## Evaluation Metrics

The predictive performance of STRCDA is evaluated using:

* Accuracy (Acc)
* F1-score (F1)
* Matthews Correlation Coefficient (MCC)
* Area Under ROC Curve (AUC)
* Area Under Precision-Recall Curve (AUPR)

---

## Results

Performance on the CircR2Disease dataset under five-fold cross-validation:

| Dataset       | AUC    | AUPR   |
| ------------- | ------ | ------ |
| CircR2Disease | 0.9771 | 0.9826 |

---

## Notes

* Some dataset-specific dimensions are hard-coded in the current implementation. Please modify them accordingly when switching datasets.
* Random initialization may cause slight fluctuations in experimental results.
* The implementation is based on TensorFlow v1.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{STRCDA2026,
  title={Inferring circRNA-Disease Associations via Sparse Topological Representation Learning and Dual-View Decoding},
  author={Liu, Chang-Chun and Wei, Meng-Meng and Lu, Mian-Shuo and Wang, Lei},
  journal={International Journal of Molecular Sciences},
  year={2026}
}
```

---

## Data Availability

The source code and datasets are publicly available at:

https://github.com/591286260/STRCDA

Further inquiries can be directed to the corresponding authors.
