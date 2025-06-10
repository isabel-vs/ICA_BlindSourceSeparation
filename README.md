# ICA_BlindSourceSeparation

This Julia package is part of a TU Berlin project on blind source separation of ECG recordings using ICA.

The goal is to separate the heartbeats of mother and foetus using different ICA algorithms. The implemented algorithms are (as of now):

- 

The used dataset is a set of cutaneous potential recordings of a pregnant woman (8 channels) from the SISTA DaISy database
- **Dataset** https://ftp.esat.kuleuven.be/pub/SISTA/data/biomedical/foetal_ecg.dat.gz
- **Description** https://ftp.esat.kuleuven.be/pub/SISTA/data/biomedical/foetal_ecg.txt

\
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Erik Felgendreher.github.io/ICA_BlindSourceSeparation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Erik Felgendreher.github.io/ICA_BlindSourceSeparation.jl/dev/)
[![Build Status](https://github.com/Erik Felgendreher/ICA_BlindSourceSeparation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Erik Felgendreher/ICA_BlindSourceSeparation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Erik Felgendreher/ICA_BlindSourceSeparation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Erik Felgendreher/ICA_BlindSourceSeparation.jl)
