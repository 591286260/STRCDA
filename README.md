# STRCDA

Sparse Topological Representation Learning and Dual-View Decoding for circRNA-Disease Association Prediction

## Abstract

Circular RNAs (circRNAs) are key regulators in the onset and progression of complex diseases, offering promise as diagnostic and prognostic biomarkers. However, most putative circRNA-disease associations remain experimentally unverified, largely due to the cost and time demands of wet-lab approaches. To bridge this gap, we present STRCDA (Sparse Topological Representation learning for CircRNA-Disease Associations). The pipeline first constructs fused similarity profiles for circRNAs and diseases by integrating diverse biological attributes. These initial matrices are then refined via random walk with restart to capture local features. Subsequently, a sparse-constrained dual-branch graph autoencoder extracts holistic topological embeddings from the refined local features and the known interaction network. Finally, an XGBoost classifier scores potential circRNA-disease pairs. On the CircR2Disease dataset, STRCDA achieves an AUC of 0.9771 and an AUPR of 0. 9826 under five-fold cross-validation. Notably, 18 of the top 20 predicted associations were confirmed by independent experimental evidence, highlighting STRCDA’s efficacy as a robust tool for uncovering circRNA function in disease.

## Keywords

circRNA-disease association; graph autoencoder; sparse representation learning; dual-view decoding; random walk with restart; graph neural network

## Framework

The STRCDA framework mainly consists of the following components:

1. Multi-source similarity construction and fusion
2. Local topology enhancement via Random Walk with Restart (RWR)
3. Dual-decoder sparse graph autoencoder
4. Sparse regularization with L1 constraint
5. XGBoost-based association prediction

## Datasets

The proposed model is evaluated on the following benchmark datasets:

* CircR2Disease
* CircAtlas
* Circ2Disease
* CircRNADisease

## Evaluation Metrics

The predictive performance of STRCDA is evaluated using:

* Accuracy (Acc)
* F1-score (F1)
* Matthews Correlation Coefficient (MCC)
* Area Under ROC Curve (AUC)
* Area Under Precision-Recall Curve (AUPR)

## Citation

If you find this work useful, please cite our paper.

```bibtex
@article{STRCDA2026,
  title={Inferring circRNA-Disease Associations via Sparse Topological Representation Learning and Dual-View Decoding},
  author={Liu, Chang-Chun and Wei, Meng-Meng and Lu, Mian-Shuo and Wang, Lei},
  journal={International Journal of Molecular Sciences},
  year={2026}
}
```
