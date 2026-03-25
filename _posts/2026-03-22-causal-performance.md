---
layout: post # needs to be post
title: Why model performance is not the same as causal performance # title of your post
featured-img: sleek #optional - if you want you can include hero image
categories: [Blog, Causal Inference]
---

Hi, this is my first post here, and I actually been longing to write about this because it involves a subject I care a lot about: Causal Inference.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 10000

# Covariates
X = np.random.normal(0, 1, n)   # confounder
Z = np.random.normal(0, 1, n)   # strong predictor of treatment (instrument-like)
W = np.random.normal(0, 1, n)   # prognostic (affects outcome only)

# Treatment assignment (depends on X and Z)
logit_p = 0.8 * X + 2.5 * Z
p = 1 / (1 + np.exp(-logit_p))
T = np.random.binomial(1, p)

# Potential outcomes
# True treatment effect = 2
epsilon = np.random.normal(0, 1, n)
Y0 = 1.5 * X + 2.0 * W + epsilon
Y1 = Y0 + 2

# Observed outcome
Y = T * Y1 + (1 - T) * Y0

df = pd.DataFrame({'Y': Y, 'T': T, 'X': X, 'Z': Z, 'W': W})
```