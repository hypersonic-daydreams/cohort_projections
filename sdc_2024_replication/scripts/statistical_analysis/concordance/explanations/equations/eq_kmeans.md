# Equation Explanation: K-Means Clustering Objective

**Number in Paper:** Eq. 11
**Category:** Unsupervised Learning
**Paper Section:** 2.6 Machine Learning Methods

---

## What This Equation Does

K-Means Clustering is an algorithm that automatically groups similar observations into clusters. Unlike regression or classification (where you're trying to predict something), K-Means is unsupervised--you're not predicting a known outcome, you're discovering structure in your data. The algorithm finds groups of observations that are similar to each other within the group but different from observations in other groups.

In the context of migration analysis, K-Means might group origin countries based on their migration patterns (size, growth, volatility), or group states based on their immigrant-receiving characteristics. The result is a set of K clusters, where each observation belongs to exactly one cluster, and each cluster has a "centroid" (center point) that represents the typical member of that cluster. The equation shown is the objective function--the quantity that K-Means tries to minimize. It measures the total "spread" of points around their cluster centers, and the algorithm seeks cluster assignments that make this spread as small as possible.

---

## The Formula

$$
\arg\min_{\mathcal{C}} \sum_{k=1}^{K} \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $\arg\min$ | "Find the value that minimizes"--we want the cluster assignments that minimize the objective | Operator | N/A |
| $\mathcal{C}$ | The set of all cluster assignments (which cluster each observation belongs to) | Decision variable | Partition of observations into $K$ groups |
| $K$ | Number of clusters (chosen by the analyst or by a criterion like silhouette score) | Constant | Positive integer (typically 2-10) |
| $C_k$ | The set of observations assigned to cluster $k$ | Set | Subset of observation indices |
| $\sum_{i \in C_k}$ | Sum over all observations $i$ that belong to cluster $k$ | Summation | N/A |
| $\mathbf{x}_i$ | Feature vector for observation $i$ | Data | Vector of real numbers (dimension $p$) |
| $\boldsymbol{\mu}_k$ | Centroid (mean) of cluster $k$ | Computed | Vector of real numbers (dimension $p$) |
| $\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$ | Squared Euclidean distance from observation $i$ to its cluster centroid | Distance measure | Non-negative real number |

---

## Step-by-Step Interpretation

1. **The centroid ($\boldsymbol{\mu}_k$):** For each cluster, the centroid is the average position of all points in that cluster. If you have a cluster of 10 countries based on migration characteristics, the centroid represents the "average" migration pattern of those 10 countries. Mathematically:
   $$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$$

2. **The squared distance ($\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$):** For each observation, we measure how far it is from its cluster's centroid. The squared Euclidean distance is used (rather than plain distance) for mathematical convenience. Points closer to their centroid contribute less to the objective.

3. **The inner sum ($\sum_{i \in C_k}$):** For each cluster, we add up the squared distances of all its members to the centroid. This gives the "within-cluster sum of squares" (WCSS) for that cluster. A tight cluster (where all members are similar) has a small WCSS.

4. **The outer sum ($\sum_{k=1}^{K}$):** We add up the WCSS across all K clusters to get the total within-cluster sum of squares. This is the objective function.

5. **The minimization ($\arg\min_{\mathcal{C}}$):** We want to find the cluster assignments that make the total WCSS as small as possible. In other words, we want clusters where points within each cluster are as close as possible to their centroid.

---

## Worked Example

**Setup:**
Suppose you want to cluster 6 origin countries based on two features: (1) average annual LPR admissions to North Dakota, and (2) growth rate of admissions. The data (after standardization) looks like this:

| Country | Admissions (standardized) | Growth Rate (standardized) |
|---------|---------------------------|---------------------------|
| A | -0.5 | -0.3 |
| B | -0.7 | -0.1 |
| C | 1.5 | 0.8 |
| D | 1.3 | 1.0 |
| E | 0.2 | -0.6 |
| F | 0.1 | -0.4 |

You decide to use K=2 clusters.

**Calculation:**
```
After running K-Means, suppose the algorithm assigns:
  Cluster 1: {A, B, E, F} -- low-admission countries with varied growth
  Cluster 2: {C, D} -- high-admission countries with high growth

Step 1: Calculate centroids
  Centroid 1 (mu_1):
    Admissions: (-0.5 + -0.7 + 0.2 + 0.1) / 4 = -0.225
    Growth: (-0.3 + -0.1 + -0.6 + -0.4) / 4 = -0.35
    mu_1 = [-0.225, -0.35]

  Centroid 2 (mu_2):
    Admissions: (1.5 + 1.3) / 2 = 1.4
    Growth: (0.8 + 1.0) / 2 = 0.9
    mu_2 = [1.4, 0.9]

Step 2: Calculate squared distances for Cluster 1
  Country A: (-0.5 - (-0.225))^2 + (-0.3 - (-0.35))^2 = 0.076 + 0.0025 = 0.0785
  Country B: (-0.7 - (-0.225))^2 + (-0.1 - (-0.35))^2 = 0.226 + 0.0625 = 0.2885
  Country E: (0.2 - (-0.225))^2 + (-0.6 - (-0.35))^2 = 0.181 + 0.0625 = 0.2435
  Country F: (0.1 - (-0.225))^2 + (-0.4 - (-0.35))^2 = 0.106 + 0.0025 = 0.1085
  WCSS_1 = 0.0785 + 0.2885 + 0.2435 + 0.1085 = 0.719

Step 3: Calculate squared distances for Cluster 2
  Country C: (1.5 - 1.4)^2 + (0.8 - 0.9)^2 = 0.01 + 0.01 = 0.02
  Country D: (1.3 - 1.4)^2 + (1.0 - 0.9)^2 = 0.01 + 0.01 = 0.02
  WCSS_2 = 0.02 + 0.02 = 0.04

Step 4: Total objective
  Total WCSS = 0.719 + 0.04 = 0.759
```

**Interpretation:**
- Cluster 1 contains countries with below-average or moderate admission levels and declining or stable growth. These might be "mature" source countries or countries with declining migration streams.
- Cluster 2 contains high-admission, high-growth countries. These might be "emerging" or "booming" source countries.
- The total WCSS of 0.759 is relatively low, suggesting the clusters are fairly tight. If we had assigned countries differently (e.g., putting E in Cluster 2), the WCSS would likely be higher.

---

## Key Assumptions

1. **Clusters are spherical:** K-Means assumes clusters are roughly spherical (round) in feature space. It doesn't work well when clusters have elongated or irregular shapes.

2. **Similar cluster sizes:** The algorithm tends to produce clusters of similar size. If your true clusters are very imbalanced (one giant cluster and several tiny ones), K-Means may not recover them well.

3. **Features are scaled appropriately:** Since K-Means uses Euclidean distance, features with larger scales will dominate the clustering. Always standardize features before clustering.

4. **K is known (or selected):** You must specify the number of clusters. If you don't know K, use methods like the silhouette score, elbow method, or gap statistic to select it.

5. **Euclidean distance is appropriate:** The algorithm uses Euclidean distance, which assumes all features contribute equally and linearly. For other distance metrics, consider alternative clustering algorithms.

---

## Common Pitfalls

- **Not standardizing features:** If one feature is measured in thousands and another in decimals, the large-scale feature will dominate. Standardize all features to have mean 0 and standard deviation 1 before clustering.

- **Arbitrary K selection:** Choosing K without justification leads to meaningless clusters. Use the silhouette score (higher is better, max = 1) or elbow method (look for a "bend" in the WCSS vs. K plot).

- **Local minima:** K-Means only finds a local minimum, not the global optimum. Different random initializations can give different results. Run the algorithm multiple times with different starting points and pick the solution with the lowest WCSS.

- **Interpreting clusters as "truth":** Clusters are summaries of your data, not ground truth. They depend on your choice of features, K, and even random initialization. Validate clusters against domain knowledge.

- **Including too many features:** With many features, distances become less meaningful (the "curse of dimensionality"). Consider dimensionality reduction (like PCA) before clustering if you have many features.

---

## Related Tests

- **Silhouette Score:** Measures how similar each point is to its own cluster compared to other clusters. Values range from -1 to 1, with higher values indicating better-defined clusters. Use this to select K.

- **Elbow Method:** Plot total WCSS against K. Look for an "elbow" where adding more clusters stops substantially reducing WCSS. The elbow suggests the appropriate K.

- **Gap Statistic:** Compares WCSS to what you'd expect from random data. The K that maximizes the "gap" between observed and expected WCSS is optimal.

- **Calinski-Harabasz Index:** Ratio of between-cluster variance to within-cluster variance. Higher values indicate better clustering.

---

## Python Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assume X is a DataFrame of features to cluster
# Example: countries with columns ['avg_admissions', 'growth_rate', 'volatility']

# Step 1: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Find optimal K using silhouette score
silhouette_scores = []
K_range = range(2, 10)  # Test K from 2 to 9

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")

# Find optimal K
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K based on silhouette: {optimal_k}")

# Step 3: Fit final model with optimal K
final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_model.fit_predict(X_scaled)

# Step 4: Analyze results
# Add cluster labels to original data
X['cluster'] = cluster_labels

# Get centroids (transform back to original scale for interpretation)
centroids_scaled = final_model.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original, columns=X.columns[:-1])
print("\nCluster Centroids (original scale):")
print(centroids_df)

# Cluster sizes
print("\nCluster Sizes:")
print(X['cluster'].value_counts().sort_index())

# WCSS (inertia)
print(f"\nTotal Within-Cluster Sum of Squares: {final_model.inertia_:.4f}")

# Step 5: Visualize (if 2D or reduced to 2D)
if X_scaled.shape[1] == 2:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
    plt.scatter(centroids_scaled[:, 0], centroids_scaled[:, 1],
                c='red', marker='X', s=200, edgecolors='black', label='Centroids')
    plt.xlabel('Feature 1 (standardized)')
    plt.ylabel('Feature 2 (standardized)')
    plt.title(f'K-Means Clustering (K={optimal_k})')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.show()

# Step 6: Profile clusters (describe typical characteristics)
print("\nCluster Profiles (mean values by cluster):")
print(X.groupby('cluster').mean())
```

---

## References

- MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297. [Original K-Means paper]

- Hartigan, J.A. & Wong, M.A. (1979). "Algorithm AS 136: A K-Means Clustering Algorithm." *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, 28(1), 100-108. [Efficient implementation]

- Rousseeuw, P.J. (1987). "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis." *Journal of Computational and Applied Mathematics*, 20, 53-65. [Silhouette score for K selection]

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 14. [Comprehensive treatment of clustering]

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 10. [Accessible introduction to unsupervised learning]
