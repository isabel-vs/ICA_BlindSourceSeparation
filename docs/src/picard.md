# Picard
## Description
Picard (Preconditioned ICA for Real Data) is an advanced algorithm for Independent Component Analysis that accelerates convergence using quasi-Newton optimization techniques. \
Unlike traditional ICA methods that rely purely on fixed-point or gradient-based updates, Picard employs preconditioning and curvature information to optimize the likelihood function more efficiently. It models the data as a linear mixture of independent, non-Gaussian sources and seeks to find the unmixing matrix by maximizing the log-likelihood under certain contrast functions. Picard integrates ideas from L-BFGS and uses Hessian approximations to handle poorly conditioned or noisy data more effectively than classic ICA approaches.

## Pros

**Fast Convergence**: Quasi-Newton updates significantly accelerate convergence, especially on large or complex datasets.

**Robust on Real Data**: Designed to handle the imperfections and variability typical of real-world signals.

**Improved Stability**: Uses preconditioning and curvature information to reduce the risk of divergence or poor local minima.

## Cons

**Higher Memory Use**: Maintains curvature information and history, which may increase memory consumption.

**Sensitive to Contrast Choice**: Performance can vary depending on the selected contrast function and source characteristics.

**Slower for Small Problems**: May be overkill for low-dimensional or well-conditioned datasets where simpler ICA methods suffice.