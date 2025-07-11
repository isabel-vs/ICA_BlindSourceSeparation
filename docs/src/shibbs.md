# Shibbs
## Description
The Shibbs algorithm is an iterative method used for blind source separation through joint approximate diagonalization of a set of covariance matrices. \
It aims to find a linear transformation that simultaneously diagonalizes these matrices, thereby recovering statistically independent source signals from observed mixtures. By applying successive Givens rotations, Shibbs progressively reduces off-diagonal elements, enhancing signal separation quality. This approach is particularly effective when sources exhibit distinct covariance structures, and its iterative nature allows refinement over multiple passes to improve accuracy. Shibbs is widely utilized in signal processing and data analysis tasks where separating mixed signals without prior information is essential.

## Pros
**Effective for Blind Source Separation**: Can separate mixed signals without prior knowledge about the sources.

**Joint Diagonalization**: Works well when multiple covariance matrices represent the data, improving separation quality.

**Iterative Refinement**: Can be run multiple times to improve accuracy.

**Relatively Simple Updates**: Uses Givens rotations, which are computationally efficient and stable.

**Adaptable**: Thresholds and iteration limits allow tuning based on data and desired precision.

## Cons
**Sensitivity to Initialization**: The starting point can affect the final output, sometimes leading to suboptimal local minima.

**Computational Cost**: For large datasets or high-dimensional data, iterative updates can be time-consuming.

**Parameter Sensitivity**: Choice of thresholds and max iterations needs careful tuning, as poor choices can slow convergence or degrade results.

**No Guarantee of Global Optimum**: Like many iterative methods, it may converge to local optima rather than the best possible solution.

**Convergence Issues**: May require multiple runs or iterations to reach a good solution, especially with noisy or ill-conditioned data.
As can be seen in the following diagram, only iteration 24 provided the correct separation.
![Demonstration of convergence issue](media/plot_4.svg)