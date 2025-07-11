# Getting started
This package provides multiple algorithms for Blind Source Separation. 
At this time the user can choose between JADE, Shibbs and Picard. 

## How to use the datatype SensorData
Key to using the package is the SensorData datatype. The user can either create a SensorData instance by loading it from disk using the `read_dataset()` function, or they can construct their own by provididing an array of timestamps of length $N$ and a corresponding matrix of size $N\times M$, where $M$ is the amount of sensors in the dataset.
When the user has created a SensorData instance, they might plot it or perform Blind Source separation using the perform_separation function.

## Code Example 
Load a dataset from disk

    x = read_dataset("data/foetal_ecg.dat")

Plot dataset

    plot_dataset(x; title="ECG Sensor Data")

Prepare JADE algorithm

    algo = Jade(2)

Prepare Shibbs algorithm

    algo = Shibbs(2, 1000, 1)

Prepare Picard algorithm

    algo = Picard(2, 3, 200, 1e-6, 1e-2, 10, false)

Run source separation. This yields the separated dataset as well as the unmixing matrix

    y, S = perform_separation(x, algo)

### Complete example:

This example plots the original data, the whitened data, as well as the results of Jade, Shibbs and Picard algorithm.

    x = read_dataset("data/foetal_ecg.dat")
    plot_dataset(x; title="ECG Sensor Data")

    x_white = whiten_dataset(x)
    plot_dataset(x_white; title="Whitened Sensor Data")

    algo = Jade(2)
    a, _ = perform_separation(x, algo)
    plot_dataset(a; title="Jade - Estimated Source Signals")

    algo = Shibbs(2, 1000, 1)
    b, _ = perform_separation(x, algo)
    plot_dataset(b; title="Shibbs - Estimated Source Signals")

    algo = Picard(2, 3, 200, 1e-6, 1e-2, 10, false)
    c, _ = perform_separation(x, algo)
    plot_dataset(c; title="Picard - Estimated Source Signals")
