# Next-Purchase-Date Prediction Model
A machine learning Next-Purchase-Date (NPD) pipeline for predicting which customers are most likely to make a repeat purchase after a specified cutoff date.
The model is trained on past transaction history, builds behavioral (RFM/dynamic) features, and outputs a ranked list of user_id with purchase probabilities.

## Problem

**Goal:** Predict which customers will make a purchase during the period `cutoff_train` → `cutoff_target` (e.g., 2025-07-12 → 2025-10-09), using only data before `cutoff_train`.

**Output:** CSV with `user_id` and `predicted_proba`.

## Tech
**Languages:** Python 3.14

**Libraries:** PyTorch, Pandas, NumPy, Seaborn, SkLearn 

## Quick Start

### In Google Colab or Jupyter
- Download `techtask_2.ipynb` and dataset
- Upload and Run in platforms

### Locally
- Download `techtask_2.py` and dataset
- Run in IDE

