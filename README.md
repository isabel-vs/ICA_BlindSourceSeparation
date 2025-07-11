# ICA_BlindSourceSeparation

This Julia package is part of a TU Berlin project on blind source separation of ECG recordings using ICA.

Currently implemented algorithms are:
- **JADE** (https://en.wikipedia.org/wiki/Joint_Approximation_Diagonalization_of_Eigen-matrices)
- **Shibbs** (https://ieeexplore.ieee.org/document/6790613)
- **Picard** (https://arxiv.org/abs/1706.08171)

For testing and demonstration purposes the algorithms are used to separate the heartbeats of a mother and foetus.

The used dataset is a set of cutaneous recordings of a potentially pregnant woman (8 channels) from the SISTA DaISy database
- **Dataset** https://ftp.esat.kuleuven.be/pub/SISTA/data/biomedical/foetal_ecg.dat.gz
- **Description** https://ftp.esat.kuleuven.be/pub/SISTA/data/biomedical/foetal_ecg.txt

\
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://isabel-vs.github.io/ICA_BlindSourceSeparation.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-pink.svg)](https://isabel-vs.github.io/ICA_BlindSourceSeparation.jl/dev/)
[![BuildStatus](https://github.com/isabel-vs/ICA_BlindSourceSeparation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/isabel-vs/ICA_BlindSourceSeparation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/isabel-vs/ICA_BlindSourceSeparation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/isabel-vs/ICA_BlindSourceSeparation.jl)

**Disclaimer:** The JADE, Shibbs and Picard algorithms are adaptations from JF Cardosos Matlab libraries.
