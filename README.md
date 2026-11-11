# Surface Code 3D-CNN Decoder under Noise Drift

> A data-driven decoder for the rotated surface code, evaluated for robustness against drifting physical noise.

---

## Overview

This project implements a **3D convolutional neural network (3D-CNN)** decoder for the surface code and systematically evaluates its robustness under **noise drift**. Built on Stim-generated rotated surface code memory experiments (`distance=3`, `rounds=3`), the pipeline maps detector events into a 3D spatio-temporal grid `(t, x, y)` as CNN input features, and predicts the logical error outcome.

---

## Workflow

### 1. Data Generation
- **100,000 shots** sampled at each of four fixed physical error rates:
  `p ∈ {0.001, 0.003, 0.005, 0.007}`
- Split **8:2** into train / test sets
- Stored as compressed `.npz` arrays

### 2. Model Training
- **Architecture**: pooling-free 3D-CNN — two `Conv3d` layers followed by a fully connected classification head
- **Loss**: `BCEWithLogitsLoss` with a positive-class weight to counter class imbalance
- **Training**: 15 epochs on the `p=0.005` dataset
- **Monitoring**: test logical error rate (LER) logged every epoch

### 3. Noise-Drift Robustness Test
- Segmented dataset with `p` drifting **linearly from 0.003 to 0.007**
  (10 segments × 10,000 shots)
- Three decoding strategies compared:
  | Strategy | Prior Knowledge |
  |---|---|
  | MWPM (fixed prior) | `p=0.005` assumed constant |
  | MWPM (oracle) | true `p` known per segment |
  | 3D-CNN | trained on `p=0.005` only |

### 4. Visualization
- Logical error rates of all three methods across drift segments
- Plotted on a **log scale**
- Highlights the CNN's robustness advantage over fixed-prior MWPM and its gap to the oracle upper bound

---

## Objective

Verify whether a **data-driven neural decoder** can generalize better than a classical **MWPM decoder** relying on a fixed prior when the physical noise parameter drifts — an important consideration for fault-tolerant decoding on real quantum hardware with **non-stationary noise**.


