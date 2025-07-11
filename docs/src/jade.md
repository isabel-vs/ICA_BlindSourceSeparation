# JADE
## Description
The JADE (Joint Approximate Diagonalization of Eigenmatrices) algorithm is a popular method for blind source separation, particularly in the context of Independent Component Analysis (ICA). \
It works by exploiting higher-order statistical moments, specifically, fourth-order cumulants, to identify a linear transformation that makes the separated signals as statistically independent as possible. JADE begins with a whitening step and proceeds by jointly diagonalizing a set of cumulant matrices. This approach is especially effective when non-Gaussianity plays a crucial role in distinguishing sources. 

## Pros

**Robustness**: Utilizes fourth-order statistics, enabling robust separation of non-Gaussian sources.

**No Prior Information Needed**: Operates in a fully blind setup without requiring knowledge of the source signals or mixing process.

**Joint Diagonalization**: Works well when multiple covariance matrices represent the data, improving separation quality.

## Cons

**Computational Complexity**: Calculation and diagonalization of fourth-order cumulant matrices can be computationally intensive.

**Memory Usage**: Storing and processing multiple cumulant matrices can require substantial memory, especially for high-dimensional data.

**Less Effective for Gaussian Sources**: Assumes non-Gaussianity. Performance degrades when sources are close to Gaussian.

**Scalability Limitations**: May struggle with very large datasets due to high cost of cumulant computation and matrix operations.
